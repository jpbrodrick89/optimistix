"""Exact second-order Newton minimisers.

This module provides:
- [`optimistix.SteihaugCGDescent`][]: truncated CG for trust-region subproblems.
- [`optimistix.AbstractNewtonMinimiser`][]: base class for exact-Hessian minimisers.
- [`optimistix.LineSearchNewton`][]: Newton with Armijo line-search globalisation.
- [`optimistix.TrustNewton`][]: Newton with classical trust-region globalisation.

These differ from quasi-Newton methods (BFGS, L-BFGS) in that they evaluate the
**exact** Hessian via JAX automatic differentiation at each accepted step, using
a [`lineax.JacobianLinearOperator`][] over `jax.grad(f)` tagged as symmetric.
This means the Hessian is never materialised: it is a lazy operator that computes
Hessian-vector products via forward-over-reverse AD on demand.

Solvers such as [`optimistix.NewtonDescent`][] will materialise the operator if
their linear solver requires a dense matrix (e.g. `lineax.Cholesky()`), while
[`optimistix.SteihaugCGDescent`][] uses Hessian-vector products directly and
never needs the full matrix.

For large-scale problems consider [`optimistix.BFGS`][] or [`optimistix.LBFGS`][].

### Comparison with the quasi-Newton solvers in optimistix

| Attribute          | BFGS / L-BFGS                | LineSearchNewton / TrustNewton  |
|--------------------|------------------------------|---------------------------------|
| Hessian            | Approximate (rank-2 update)  | Exact (FunctionLinearOperator)  |
| Cost per HVP       | O(n)                         | O(cost of one grad eval)        |
| Non-convex support | Limited (requires PD approx) | TrustNewton + SteihaugCGDescent |
| Good for           | Large-scale problems         | Small/medium, accurate solution |
"""

from collections.abc import Callable
from typing import Any, cast, Generic

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox import AbstractVar
from equinox.internal import ω
from jaxtyping import Array, Bool, Int, PyTree, Scalar

from .._custom_types import Aux, DescentState, Fn, SearchState, Y
from .._minimise import AbstractMinimiser
from .._misc import (
    cauchy_termination,
    default_verbose,
    filter_cond,
    max_norm,
    tree_dot,
    tree_full_like,
    tree_where,
    two_norm,
)
from .._search import AbstractDescent, AbstractSearch, FunctionInfo
from .._solution import RESULTS
from .backtracking import BacktrackingArmijo
from .gauss_newton import NewtonDescent
from .trust_region import ClassicalTrustRegion


# Stable eqx.Module wrapper for ∇f; cached in state so jax.linearize sees the
# same jaxpr structure every step (needed for the normalisation trick below).
class _HessianGradFn(eqx.Module):
    fn: Callable
    args: Any

    def __call__(self, y: Y) -> Y:
        return jax.grad(lambda _y: self.fn(_y, self.args)[0])(y)


def _make_hessian_f_info(
    hessian_grad_fn: _HessianGradFn,
    fn: Fn[Y, Scalar, Aux],
    y: Y,
    args: PyTree,
    tags: frozenset[object],
) -> tuple[FunctionInfo.EvalGradHessian, Aux]:
    f_val, aux = fn(y, args)
    grad, hess_mv_fn = jax.linearize(hessian_grad_fn, y)
    hessian = lx.FunctionLinearOperator(
        hess_mv_fn, jax.eval_shape(lambda: y), frozenset({lx.symmetric_tag}) | tags
    )
    return FunctionInfo.EvalGradHessian(f_val, grad, hessian), aux


# ---------------------------------------------------------------------------
# SteihaugCGDescent
# ---------------------------------------------------------------------------


class _SteihaugCGDescentState(eqx.Module, Generic[Y]):
    f_info: FunctionInfo.EvalGradHessian
    grad_norm: Scalar  # ||g||, used to scale the trust-region radius


class SteihaugCGDescent(
    AbstractDescent[
        Y,
        FunctionInfo.EvalGradHessian,
        _SteihaugCGDescentState,
    ],
):
    """Steihaug-Toint truncated CG for trust-region subproblems.

    Approximately solves the trust-region subproblem

    ```
    min  g^T p + 0.5 p^T H p
    s.t. ||p|| <= Δ
    ```

    using conjugate gradients that terminate early when:

    1. **Negative curvature**: `d^T H d ≤ 0`. The current search direction is
       extended to the trust-region boundary and returned.
    2. **Boundary hit**: the unconstrained CG step would leave the trust region.
       The step is projected back onto the boundary and returned.
    3. **CG convergence**: `||r_j|| < rtol * ||g||`. The current iterate is
       returned.

    Hessian-vector products are computed lazily via forward-over-reverse AD
    (`jax.jvp` of the gradient), so the full Hessian is never materialised.
    This makes the per-step cost O(k * cost_of_grad) for k CG iterations rather
    than O(n²).

    Because no Cholesky factorisation is required and negative curvature is
    handled gracefully, this descent is suitable for **non-convex** problems
    where the Hessian may be indefinite.

    Designed for use with [`optimistix.TrustNewton`][].
    """

    max_steps: int = 100
    rtol: float = 0.5

    def init(
        self,
        y: Y,
        f_info_struct: FunctionInfo.EvalGradHessian,
    ) -> _SteihaugCGDescentState:
        f_info_init = tree_full_like(f_info_struct, 0, allow_static=True)
        return _SteihaugCGDescentState(f_info=f_info_init, grad_norm=jnp.array(0.0))

    def query(
        self,
        y: Y,
        f_info: FunctionInfo.EvalGradHessian,
        state: _SteihaugCGDescentState,
    ) -> _SteihaugCGDescentState:
        del y, state
        return _SteihaugCGDescentState(f_info=f_info, grad_norm=two_norm(f_info.grad))

    def step(
        self, step_size: Scalar, state: _SteihaugCGDescentState
    ) -> tuple[Y, RESULTS]:
        g = state.f_info.grad
        H = state.f_info.hessian

        delta_sq = (state.grad_norm * step_size) ** 2
        r0_norm_sq = cast(Array, tree_dot(g, g))
        tol_sq = (self.rtol**2) * r0_norm_sq

        p0 = tree_full_like(g, 0)
        r0 = g
        d0 = jtu.tree_map(jnp.negative, g)

        class _CGState(eqx.Module):
            p: Any
            r: Any
            d: Any
            rr: Scalar
            result_p: Any  # committed output once done
            done: Bool[Array, ""]

        def _find_boundary_tau(p, d):
            """Find τ ≥ 0 such that ||p + τ d||² = delta_sq."""
            pd = tree_dot(p, d)
            dd = tree_dot(d, d)
            pp = tree_dot(p, p)
            disc = pd**2 - dd * (pp - delta_sq)
            safe_dd = jnp.where(dd > jnp.finfo(dd.dtype).eps, dd, 1.0)
            tau = (-pd + jnp.sqrt(jnp.maximum(disc, 0.0))) / safe_dd
            return jnp.maximum(tau, 0.0)

        def body_fn(i, cg_state: _CGState) -> _CGState:
            Hd = H.mv(cg_state.d)
            dHd = cast(Array, tree_dot(cg_state.d, Hd))

            # Case 1: negative or zero curvature — step to boundary along d.
            neg_curve = dHd <= jnp.finfo(dHd.dtype).eps
            tau_neg = _find_boundary_tau(cg_state.p, cg_state.d)
            result_neg = (cg_state.p**ω + tau_neg * cg_state.d**ω).ω

            safe_dHd = jnp.where(neg_curve, 1.0, dHd)
            alpha = cg_state.rr / safe_dHd

            p_new = (cg_state.p**ω + alpha * cg_state.d**ω).ω
            p_new_norm_sq = tree_dot(p_new, p_new)

            # Case 2: unconstrained step exits trust region — project to boundary.
            past_boundary = p_new_norm_sq >= delta_sq
            tau_bdy = _find_boundary_tau(cg_state.p, cg_state.d)
            result_bdy = (cg_state.p**ω + tau_bdy * cg_state.d**ω).ω

            r_new = (cg_state.r**ω + alpha * Hd**ω).ω
            rr_new = cast(Array, tree_dot(r_new, r_new))

            # Case 3: CG residual small enough — return current p_new.
            converged = rr_new < tol_sq

            done_now = neg_curve | past_boundary | converged
            result_now = tree_where(
                neg_curve,
                result_neg,
                tree_where(past_boundary, result_bdy, p_new),
            )

            safe_rr = jnp.where(
                cg_state.rr > jnp.finfo(cg_state.rr.dtype).eps, cg_state.rr, 1.0
            )
            beta = rr_new / safe_rr
            d_new = (-(r_new**ω) + beta * cg_state.d**ω).ω

            new_done = cg_state.done | done_now
            # Freeze result_p once we first commit; keep updating p otherwise.
            new_result_p = tree_where(
                cg_state.done,
                cg_state.result_p,
                tree_where(done_now, result_now, p_new),
            )
            new_p = tree_where(new_done, cg_state.p, p_new)
            new_r = tree_where(new_done, cg_state.r, r_new)
            new_rr = jnp.where(new_done, cg_state.rr, rr_new)
            new_d = tree_where(new_done, cg_state.d, d_new)

            return _CGState(
                p=new_p,
                r=new_r,
                d=new_d,
                rr=new_rr,
                result_p=new_result_p,
                done=new_done,
            )

        init_cg = _CGState(
            p=p0,
            r=r0,
            d=d0,
            rr=r0_norm_sq,
            result_p=p0,
            done=jnp.array(False),
        )

        final_cg = lax.fori_loop(0, self.max_steps, body_fn, init_cg)

        return final_cg.result_p, RESULTS.successful


SteihaugCGDescent.__init__.__doc__ = """**Arguments:**

- `max_steps`: Maximum number of CG iterations. Defaults to 100. A larger value
    gives a more accurate solution to the trust-region subproblem at the cost of
    more Hessian-vector products per outer step.
- `rtol`: Relative tolerance for CG convergence. CG terminates when
    `||r_j|| < rtol * ||g||`. Defaults to 0.5, which corresponds to an inexact
    Newton step (the "forcing sequence" approach).
"""


# ---------------------------------------------------------------------------
# AbstractNewtonMinimiser
# ---------------------------------------------------------------------------


class _NewtonMinimiserState(eqx.Module, Generic[Y, Aux, SearchState, DescentState]):
    hessian_grad_fn: _HessianGradFn
    # Updated every search step
    first_step: Bool[Array, ""]
    y_eval: Y
    search_state: SearchState
    # Updated after each accepted descent step
    f_info: FunctionInfo.EvalGradHessian
    aux: Aux
    descent_state: DescentState
    # Termination
    terminate: Bool[Array, ""]
    result: RESULTS
    # Used in compat.py
    num_accepted_steps: Int[Array, ""]


class AbstractNewtonMinimiser(
    AbstractMinimiser[Y, Aux, _NewtonMinimiserState],
    Generic[Y, Aux],
):
    """Abstract base class for exact second-order Newton minimisers.

    Subclasses evaluate the **exact Hessian** at each accepted step using a
    [`lineax.JacobianLinearOperator`][] wrapping `jax.grad(fn)`, tagged as
    symmetric. The Hessian is never materialised: Hessian-vector products are
    computed lazily via forward-over-reverse AD. If the chosen linear solver
    (e.g. `lineax.Cholesky()`) needs a dense matrix it will materialise on
    demand, but [`optimistix.SteihaugCGDescent`][] avoids this entirely.

    Subclasses must provide the following attributes:

    - `rtol: float`
    - `atol: float`
    - `norm: Callable[[PyTree], Scalar]`
    - `descent: AbstractDescent[Y, FunctionInfo.EvalGradHessian, Any]`
    - `search: AbstractSearch[Y, FunctionInfo.EvalGradHessian, FunctionInfo.Eval, Any]`
    - `verbose: Callable[..., None]`

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation
        to compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to
        `"bwd"`.
    """

    rtol: AbstractVar[float]
    atol: AbstractVar[float]
    norm: AbstractVar[Callable[[PyTree], Scalar]]
    descent: AbstractVar[AbstractDescent[Y, FunctionInfo.EvalGradHessian, Any]]
    search: AbstractVar[
        AbstractSearch[Y, FunctionInfo.EvalGradHessian, FunctionInfo.Eval, Any]
    ]
    verbose: AbstractVar[Callable[..., None]]

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: jax.ShapeDtypeStruct,
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _NewtonMinimiserState:
        hessian_grad_fn = _HessianGradFn(fn=fn, args=args)
        f_info_struct, _ = eqx.filter_eval_shape(
            _make_hessian_f_info, hessian_grad_fn, fn, y, args, tags
        )
        f_info = tree_full_like(f_info_struct, 0, allow_static=True)
        return _NewtonMinimiserState(
            hessian_grad_fn=hessian_grad_fn,
            first_step=jnp.array(True),
            y_eval=y,
            search_state=self.search.init(y, f_info_struct),
            f_info=f_info,
            aux=tree_full_like(aux_struct, 0),
            descent_state=self.descent.init(y, f_info_struct),
            terminate=jnp.array(False),
            result=RESULTS.successful,
            num_accepted_steps=jnp.array(0),
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _NewtonMinimiserState,
        tags: frozenset[object],
    ) -> tuple[Y, _NewtonMinimiserState, Aux]:
        f_eval, aux_eval = fn(state.y_eval, args)

        step_size, accept, search_result, search_state = self.search.step(
            state.first_step,
            y,
            state.y_eval,
            state.f_info,
            FunctionInfo.Eval(f_eval),
            state.search_state,
        )

        def accepted(descent_state):
            # Primal of linearizing ∇f is the gradient itself; same pattern as
            # Newton root finder using jax.linearize(fn, y) to get f_eval.
            grad, hess_mv_fn = jax.linearize(state.hessian_grad_fn, state.y_eval)
            hessian = lx.FunctionLinearOperator(
                hess_mv_fn,
                jax.eval_shape(lambda: state.y_eval),
                frozenset({lx.symmetric_tag}) | tags,
            )
            # Swap in residuals from this step while keeping the static jaxpr
            # from state, so filter_cond sees identical treedefs in both branches.
            dynamic = eqx.filter(hessian, eqx.is_array)
            static = eqx.filter(state.f_info.hessian, eqx.is_array, inverse=True)
            hessian = eqx.combine(dynamic, static)

            f_eval_info = FunctionInfo.EvalGradHessian(f_eval, grad, hessian)
            descent_state = self.descent.query(state.y_eval, f_eval_info, descent_state)

            y_diff = (state.y_eval**ω - y**ω).ω
            f_diff = (f_eval**ω - state.f_info.f**ω).ω
            terminate = cauchy_termination(
                self.rtol, self.atol, self.norm, state.y_eval, y_diff, f_eval, f_diff
            )
            terminate = jnp.where(state.first_step, jnp.array(False), terminate)
            return (
                state.y_eval,
                f_eval_info,
                aux_eval,
                descent_state,
                terminate,
            )

        def rejected(descent_state):
            return (
                y,
                state.f_info,
                state.aux,
                descent_state,
                jnp.array(False),
            )

        y, f_info, aux, descent_state, terminate = filter_cond(
            accept, accepted, rejected, state.descent_state
        )

        self.verbose(
            loss_this_step=("Loss on this step", f_eval),
            loss_last_accepted_step=("Loss on the last accepted step", state.f_info.f),
            step_size=("Step size", step_size),
            y=("y", state.y_eval),
            y_last_accepted_step=("y on the last accepted step", y),
        )

        y_descent, descent_result = self.descent.step(step_size, descent_state)
        y_eval = (y**ω + y_descent**ω).ω
        result = RESULTS.where(
            search_result == RESULTS.successful, descent_result, search_result
        )

        prev_aux = tree_where(state.first_step, aux, state.aux)
        state = _NewtonMinimiserState(
            hessian_grad_fn=state.hessian_grad_fn,
            first_step=jnp.array(False),
            y_eval=y_eval,
            search_state=search_state,
            f_info=f_info,
            aux=aux,
            descent_state=descent_state,
            terminate=terminate,
            result=result,
            num_accepted_steps=state.num_accepted_steps + jnp.where(accept, 1, 0),
        )
        return y, state, prev_aux

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _NewtonMinimiserState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return state.terminate, state.result

    def postprocess(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _NewtonMinimiserState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}


# ---------------------------------------------------------------------------
# LineSearchNewton
# ---------------------------------------------------------------------------


class LineSearchNewton(AbstractNewtonMinimiser[Y, Aux]):
    """Newton minimiser with Armijo backtracking line-search globalisation.

    At each accepted step an exact Hessian-vector-product operator is built via
    `jax.grad` and [`lineax.JacobianLinearOperator`][], then the Newton system
    `H δ = -g` is solved to obtain the search direction. An Armijo backtracking
    line search then finds an acceptable step length.

    This is a good choice for **convex or near-convex** problems where the
    Hessian is guaranteed to be (or very close to) positive definite. For
    non-convex problems where the Hessian can be indefinite, prefer
    [`optimistix.TrustNewton`][] with [`optimistix.SteihaugCGDescent`][].

    Supports the following `options`:

    - `autodiff_mode`: `"fwd"` or `"bwd"`. Defaults to `"bwd"`.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    descent: NewtonDescent
    search: BacktrackingArmijo
    verbose: Callable[..., None]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None),
        verbose: bool | Callable[..., None] = False,
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = NewtonDescent(linear_solver=linear_solver)
        self.search = BacktrackingArmijo()
        self.verbose = default_verbose(verbose)


LineSearchNewton.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `linear_solver`: The linear solver used to solve `H δ = -g`. Defaults to
    `lineax.AutoLinearSolver(well_posed=None)`. For problems where the Hessian
    is known to be positive definite, `lineax.Cholesky()` is faster.
- `verbose`: Whether to print out extra information about how the solve is
    proceeding. Can be `False`, `True`, or a callable `**kwargs -> None`.
"""


# ---------------------------------------------------------------------------
# TrustNewton
# ---------------------------------------------------------------------------


class TrustNewton(AbstractNewtonMinimiser[Y, Aux]):
    """Newton minimiser with classical trust-region globalisation.

    At each accepted step an exact Hessian-vector-product operator is built via
    `jax.grad` and [`lineax.JacobianLinearOperator`][], then the trust-region
    subproblem is solved. Two descent directions are supported:

    - **`NewtonDescent`** (default): solves the full Newton system and scales the
      step to fit within the trust region. Works well for convex problems.
    - **`SteihaugCGDescent`**: solves the trust-region subproblem approximately
      via truncated CG. Handles indefinite Hessians gracefully, making it
      suitable for **non-convex** problems. Never materialises the Hessian.

    To use `SteihaugCGDescent`, pass `use_steihaug=True`.

    Supports the following `options`:

    - `autodiff_mode`: `"fwd"` or `"bwd"`. Defaults to `"bwd"`.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    descent: NewtonDescent | SteihaugCGDescent
    search: ClassicalTrustRegion
    verbose: Callable[..., None]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None),
        use_steihaug: bool = False,
        steihaug_max_steps: int = 100,
        verbose: bool | Callable[..., None] = False,
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        if use_steihaug:
            self.descent = SteihaugCGDescent(max_steps=steihaug_max_steps)
        else:
            self.descent = NewtonDescent(linear_solver=linear_solver)
        self.search = ClassicalTrustRegion()
        self.verbose = default_verbose(verbose)


TrustNewton.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `linear_solver`: The linear solver used to solve the Newton system when
    `use_steihaug=False`. Ignored when `use_steihaug=True`. Defaults to
    `lineax.AutoLinearSolver(well_posed=None)`.
- `use_steihaug`: If `True`, use [`optimistix.SteihaugCGDescent`][] to solve
    the trust-region subproblem via truncated CG. This handles indefinite
    Hessians and is recommended for non-convex problems.
- `steihaug_max_steps`: Maximum CG iterations per outer step when
    `use_steihaug=True`. Defaults to 100.
- `verbose`: Whether to print out extra information about how the solve is
    proceeding. Can be `False`, `True`, or a callable `**kwargs -> None`.
"""
