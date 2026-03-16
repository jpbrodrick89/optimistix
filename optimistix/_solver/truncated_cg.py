"""Truncated CG linear solver for indefinite operators (Newton-CG).

Started from a copy of lineax's CG implementation and modified to:
- remove the positive-definite operator requirement
- exit early on negative curvature instead of injecting NaN
- optionally exit on a trust-region boundary crossing, with projection
- support per-call rtol override (Eisenstat-Walker hook)
"""

from typing import Any, TypeAlias, cast

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar
from lineax._norm import max_norm as _lx_max_norm
from lineax._solution import RESULTS as _lxRESULTS

from .._misc import tree_dot, tree_full_like, tree_where


def _find_boundary_tau(p: Any, d: Any, delta_sq: Scalar) -> Scalar:
    """Find τ ≥ 0 such that ‖p + τ d‖² = delta_sq.

    Solves the positive root of the quadratic ‖p + τd‖² = Δ²,
    i.e. τ = (-p·d + sqrt((p·d)² - ‖d‖²(‖p‖² - Δ²))) / ‖d‖²
    """
    pd = cast(Scalar, tree_dot(p, d))
    dd = cast(Scalar, tree_dot(d, d))
    pp = cast(Scalar, tree_dot(p, p))
    disc = pd**2 - dd * (pp - delta_sq)
    safe_dd = jnp.where(dd > jnp.finfo(dd.dtype).eps, dd, 1.0)
    tau = (-pd + jnp.sqrt(jnp.maximum(disc, 0.0))) / safe_dd
    return jnp.maximum(tau, 0.0)


_TruncatedCGState: TypeAlias = lx.AbstractLinearOperator


class TruncatedCG(lx.AbstractLinearSolver[_TruncatedCGState]):
    """CG solver for potentially indefinite operators, for use in Newton-CG.

    Unlike `lineax.CG`, this solver does not require a positive-definite operator.
    When negative curvature is detected (`d^T H d ≤ 0`), the solver exits early and
    returns the current partial CG iterate as the descent direction. This is the
    truncation strategy used by scipy's Newton-CG method.

    On the first CG step, if negative curvature is detected before any progress has
    been made (iterate is still zero), the initial residual direction (equal to the
    negative gradient for Newton problems) is returned as a steepest-descent fallback.

    Supports the following `options` (passed to `lx.linear_solve(..., options=...)`):

    - `"rtol"`: Override the solver's `rtol` for this call. Intended for the
        Eisenstat-Walker tolerance schedule in Newton-CG, where the tolerance is
        tightened as the outer iterate approaches the solution.
    - `"delta"`: Trust-region radius. If provided, the solver exits early when
        `‖y‖ ≥ delta`, projects the step onto the trust-region boundary (solving
        `‖p + τ d‖ = Δ`), and returns the projected step as `.value`. Negative
        curvature exits are also projected onto the boundary when `delta` is finite.
        Defaults to `jnp.inf` (no trust-region constraint, no projection).

    - `y0`: Initial estimate of the solution. Defaults to all zeros.
    """

    rtol: float
    atol: float
    norm: Any = _lx_max_norm
    stabilise_every: int | None = 10
    max_steps: int | None = None

    def __check_init__(self):
        if isinstance(self.rtol, (int, float)) and self.rtol < 0:
            raise ValueError("`TruncatedCG` requires `rtol >= 0`.")
        if isinstance(self.atol, (int, float)) and self.atol < 0:
            raise ValueError("`TruncatedCG` requires `atol >= 0`.")
        if (
            isinstance(self.atol, (int, float))
            and isinstance(self.rtol, (int, float))
            and self.atol == 0
            and self.rtol == 0
            and self.max_steps is None
        ):
            raise ValueError(
                "Must specify `rtol`, `atol`, or `max_steps` (or some combination "
                "of all three)."
            )

    def init(
        self, operator: lx.AbstractLinearOperator, options: dict[str, Any]
    ) -> _TruncatedCGState:
        del options
        # Unlike lineax.CG we do not require positive or negative semidefiniteness:
        # TruncatedCG is designed to handle indefinite operators.
        if not eqx.tree_equal(operator.in_structure(), operator.out_structure()):
            raise ValueError(
                "`TruncatedCG` may only be used for square linear operators."
            )
        return lx.linearise(operator)

    def compute(
        self,
        state: _TruncatedCGState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], _lxRESULTS, dict[str, Any]]:
        operator = state

        # Per-call overrides (Eisenstat-Walker, trust-region radius).
        rtol = options.get("rtol", self.rtol)
        atol = options.get("atol", self.atol)
        delta = options.get("delta", jnp.inf)
        delta_sq = delta**2

        y0 = options.get("y0", tree_full_like(vector, 0))

        leaves, _ = jtu.tree_flatten(vector)
        size = sum(leaf.size for leaf in leaves)
        if self.max_steps is None:
            max_steps = 10 * size  # match lineax.CG default
        else:
            max_steps = self.max_steps

        r0 = (vector**ω - operator.mv(y0) ** ω).ω
        d0 = r0  # initial CG direction = initial residual (standard CG)
        gamma0 = cast(Scalar, tree_dot(r0, r0))

        has_scale = not (
            isinstance(atol, (int, float))
            and isinstance(rtol, (int, float))
            and atol == 0
            and rtol == 0
        )
        if has_scale:
            b_scale = (atol + rtol * ω(vector).call(jnp.abs)).ω

        def not_converged(r, diff, y):
            if has_scale:
                with jax.numpy_dtype_promotion("standard"):
                    y_scale = (atol + rtol * ω(y).call(jnp.abs)).ω
                    norm1 = self.norm((r**ω / b_scale**ω).ω)  # pyright: ignore
                    norm2 = self.norm((diff**ω / y_scale**ω).ω)
                return (norm1 > 1) | (norm2 > 1)
            else:
                return jnp.array(True)

        class _CGState(eqx.Module):
            diff: Any       # last step (for convergence check)
            y: Any          # current iterate
            r: Any          # current residual
            d: Any          # current CG direction
            gamma: Scalar   # r^T r
            step: Scalar
            done: Bool[Array, ""]           # neg_curv or boundary exit fired
            neg_curv: Bool[Array, ""]       # negative curvature exit
            hit_boundary: Bool[Array, ""]   # trust-region boundary exit
            result_y: Any   # iterate to return at exit (y_before for early exits)
            result_d: Any   # CG direction at exit (for boundary projection)

        def cond_fun(cg_state: _CGState) -> Bool[Array, ""]:
            out = cg_state.gamma > 0
            out = out & (cg_state.step < max_steps)
            out = out & not_converged(cg_state.r, cg_state.diff, cg_state.y)
            out = out & ~cg_state.done
            return out

        def body_fun(cg_state: _CGState) -> _CGState:
            mat_d = operator.mv(cg_state.d)
            inner_prod = cast(Scalar, tree_dot(mat_d, cg_state.d))

            # --- Departure from lineax.CG ---
            # lineax.CG injects NaN when inner_prod is near-zero (lines 174-178).
            # Instead we detect negative curvature explicitly and exit cleanly.
            neg_curv = inner_prod <= jnp.finfo(jnp.result_type(*leaves)).eps
            # Guard against division issues in the NaN-free path.
            safe_inner_prod = jnp.where(neg_curv, 1.0, inner_prod)

            alpha = cg_state.gamma / safe_inner_prod
            diff = (alpha * cg_state.d**ω).ω
            y_new = (cg_state.y**ω + diff**ω).ω

            # --- Boundary exit (trust-region use) ---
            y_new_norm_sq = cast(Scalar, tree_dot(y_new, y_new))
            hit_boundary = y_new_norm_sq >= delta_sq

            # Residual update — same periodic stabilisation as lineax.CG.
            def stable_r():
                return (vector**ω - operator.mv(y_new) ** ω).ω

            def cheap_r():
                return (cg_state.r**ω - alpha * mat_d**ω).ω

            if self.stabilise_every == 1:
                r_new = stable_r()
            elif self.stabilise_every is None:
                r_new = cheap_r()
            else:
                stable_step = (
                    eqx.internal.unvmap_max(cg_state.step) % self.stabilise_every
                ) == 0
                stable_step = eqx.internal.nonbatchable(stable_step)
                r_new = lax.cond(stable_step, stable_r, cheap_r)

            gamma_new = cast(Scalar, tree_dot(r_new, r_new))
            safe_gamma = jnp.where(
                cg_state.gamma > jnp.finfo(gamma_new.dtype).eps, cg_state.gamma, 1.0
            )
            beta = gamma_new / safe_gamma
            d_new = (r_new**ω + beta * cg_state.d**ω).ω

            # Determine what to return if we exit here.
            #
            # Trust-region mode (delta finite): project onto the boundary sphere
            # by solving ‖y_before + τ d‖ = Δ, for both neg_curv and boundary exits.
            #
            # Line-search mode (delta=inf): no projection.  On neg_curv at the very
            # first step (y=0) fall back to d (= r0 = -g) so the Armijo search has
            # a non-trivial descent direction; on later steps return y_before.
            done_now = neg_curv | hit_boundary
            should_project = done_now & ~jnp.isinf(delta)

            y_is_zero = cast(Scalar, tree_dot(cg_state.y, cg_state.y)) <= 0
            first_step_linesearch_fallback = y_is_zero & neg_curv & jnp.isinf(delta)
            y_before = tree_where(first_step_linesearch_fallback, cg_state.d, cg_state.y)

            tau = _find_boundary_tau(cg_state.y, cg_state.d, delta_sq)
            y_projected = (cg_state.y**ω + tau * cg_state.d**ω).ω

            y_exit = tree_where(done_now, y_before, y_new)
            result_y_now = tree_where(should_project, y_projected, y_exit)

            return _CGState(
                diff=diff,
                y=y_new,
                r=r_new,
                d=d_new,
                gamma=gamma_new,
                step=cg_state.step + 1,
                done=done_now,
                neg_curv=neg_curv,
                hit_boundary=hit_boundary,
                result_y=result_y_now,
                result_d=cg_state.d,
            )

        # Initialise result_y to r0 (= -g for Newton-CG) so that if max_steps=0
        # or negative curvature fires on the very first step (y=0), we return the
        # steepest-descent direction rather than a zero vector.
        init_state = _CGState(
            diff=ω(y0).call(lambda x: jnp.full_like(x, jnp.inf)).ω,
            y=y0,
            r=r0,
            d=d0,
            gamma=gamma0,
            step=jnp.array(0),
            done=jnp.array(False),
            neg_curv=jnp.array(False),
            hit_boundary=jnp.array(False),
            result_y=r0,
            result_d=d0,
        )

        final = lax.while_loop(cond_fun, body_fun, init_state)

        # If the loop exited via done (neg_curv or boundary), or via convergence
        # (not_converged became False), result_y is set correctly by the last
        # body_fun call. If max_steps was reached, result_y is the last y_new.
        solution = final.result_y

        if self.max_steps is None:
            result = _lxRESULTS.where(
                final.step == max_steps, _lxRESULTS.singular, _lxRESULTS.successful
            )
        elif has_scale:
            result = _lxRESULTS.where(
                final.step == max_steps,
                _lxRESULTS.max_steps_reached,
                _lxRESULTS.successful,
            )
        else:
            result = _lxRESULTS.successful

        stats = {
            "num_steps": final.step,
            "max_steps": jnp.array(max_steps),
            "negative_curvature": final.neg_curv,
            "hit_boundary": final.hit_boundary,
            "direction": final.result_d,
        }
        return solution, result, stats

    def transpose(
        self, state: _TruncatedCGState, options: dict[str, Any]
    ) -> tuple[_TruncatedCGState, dict[str, Any]]:
        return state.transpose(), options

    def conj(
        self, state: _TruncatedCGState, options: dict[str, Any]
    ) -> tuple[_TruncatedCGState, dict[str, Any]]:
        return lx.conj(state), options

    def assume_full_rank(self) -> bool:
        return False


TruncatedCG.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for CG convergence. Can be overridden per-call via
    `options["rtol"]` (e.g. for Eisenstat-Walker tolerance scheduling).
- `atol`: Absolute tolerance for CG convergence. Can be overridden per-call via
    `options["atol"]`.
- `norm`: The norm to use for convergence checking. Defaults to `lineax.max_norm`.
- `stabilise_every`: Recompute the residual exactly every this many steps to
    correct floating-point drift. Matches `lineax.CG` default of 10.
- `max_steps`: Maximum number of CG iterations. Defaults to `10 * n` (matching
    `lineax.CG`). At max_steps the solve returns `RESULTS.singular` (if
    `max_steps` was inferred) or `RESULTS.max_steps_reached` (if set explicitly).
"""
