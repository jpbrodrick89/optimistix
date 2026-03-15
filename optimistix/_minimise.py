from typing import Any, cast, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox import AbstractVar
from jaxtyping import PyTree, Scalar

from ._adjoint import AbstractAdjoint, ImplicitAdjoint
from ._custom_types import Aux, Fn, MaybeAuxFn, SolverState, Y
from ._iterate import AbstractIterativeSolver, iterative_solve
from ._misc import inexact_asarray, NoneAux, OutAsArray, tree_full_like
from ._solution import Solution


if TYPE_CHECKING:
    from ._root_find import AbstractRootFinder


class AbstractMinimiser(AbstractIterativeSolver[Y, Scalar, Aux, SolverState]):
    """Abstract base class for all minimisers."""


def _rewrite_fn(minimum, _, inputs):
    minimise_fn, _, _, args, *_ = inputs
    del inputs

    def min_no_aux(x):
        f_val, _ = minimise_fn(x, args)
        return f_val

    return jax.grad(min_no_aux)(minimum)


# Keep `optx.implicit_jvp` happy.
# https://github.com/patrick-kidger/optimistix/issues/102#event-15786001854
if _rewrite_fn.__globals__["__name__"].startswith("jaxtyping"):
    _rewrite_fn = _rewrite_fn.__wrapped__  # pyright: ignore[reportFunctionMemberAccess]


def _to_grad_fn(fn, y, args):
    # fn(y, args) = (scalar, aux) after NoneAux/OutAsArray wrapping.
    # Returns (grad, (grad, aux)) — grad is placed in both the root-function output
    # position AND the aux, so that _RootToMinimise can track it for termination.
    # This mirrors _to_minimise_fn in _root_find.py which returns (norm(root), (root, aux)).
    (_, aux), grad = jax.value_and_grad(lambda _y: fn(_y, args), has_aux=True)(y)
    return grad, (grad, aux)


# Adds an additional termination condition that `∇f(y)` is near zero.
# Analogous to `_MinimToRoot` in `_root_find.py`, which adds `||f(y)|| < atol`
# when a minimiser is used as a root finder.
class _RootToMinimise(AbstractIterativeSolver):
    solver: AbstractVar[AbstractIterativeSolver]

    @property
    def rtol(self):
        return self.solver.rtol

    @property
    def atol(self):
        return self.solver.atol

    @property
    def norm(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.solver.norm

    def init(self, fn, y, args, options, f_struct, aux_struct, tags):
        grad_struct, _ = aux_struct
        init_state = self.solver.init(fn, y, args, options, f_struct, aux_struct, tags)
        grad_inf = tree_full_like(grad_struct, jnp.inf)
        return (init_state, grad_inf)

    def step(self, fn, y, args, options, state, tags):
        state, _ = state
        new_y, new_state, (grad, aux) = self.solver.step(
            fn, y, args, options, state, tags
        )
        return new_y, (new_state, grad), (grad, aux)

    def terminate(self, fn, y, args, options, state, tags):
        state, grad = state
        terminate, result = self.solver.terminate(fn, y, args, options, state, tags)
        # No rtol, because `rtol * 0 = 0`.
        near_zero = self.norm(grad) < self.atol
        return terminate & near_zero, result

    def postprocess(self, fn, y, aux, args, options, state, tags, result):
        state, _ = state
        return self.solver.postprocess(fn, y, aux, args, options, state, tags, result)


class _ConcreteRootToMinimise(_RootToMinimise):
    solver: AbstractIterativeSolver

    # Redeclare these three to work around the Equinox bug fixed here:
    # https://github.com/patrick-kidger/equinox/pull/544
    @property
    def rtol(self):
        return self.solver.rtol

    @property
    def atol(self):
        return self.solver.atol

    @property
    def norm(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.solver.norm


@eqx.filter_jit
def minimise(
    fn: MaybeAuxFn[Y, Scalar, Aux],
    # no type parameters, see https://github.com/microsoft/pyright/discussions/5599
    solver: "AbstractMinimiser | AbstractRootFinder",
    y0: Y,
    args: PyTree[Any] = None,
    options: dict[str, Any] | None = None,
    *,
    has_aux: bool = False,
    max_steps: int | None = 256,
    adjoint: AbstractAdjoint = ImplicitAdjoint(),
    throw: bool = True,
    tags: frozenset[object] = frozenset(),
) -> Solution[Y, Aux]:
    """Minimise a function.

    This minimises a nonlinear function `fn(y, args)` which returns a scalar value.

    **Arguments:**

    - `fn`: The objective function. This should take two arguments: `fn(y, args)` and
        return a scalar.
    - `solver`: The solver to use. This should be an
        [`optimistix.AbstractMinimiser`][] or an
        [`optimistix.AbstractRootFinder`][]. If it is a root finder, the
        minimisation is performed by finding the roots of the gradient `∇f(y) = 0`,
        enabling the use of true second-order methods (e.g.
        [`optimistix.Newton`][] with exact Hessian-vector products via JAX AD).
    - `y0`: An initial guess for what `y` may be.
    - `args`: Passed as the `args` of `fn(y, args)`.
    - `options`: Individual solvers may accept additional runtime arguments.
        See each individual solver's documentation for more details.
    - `has_aux`: If `True`, then `fn` may return a pair, where the first element is its
        function value, and the second is just auxiliary data. Keyword only argument.
    - `max_steps`: The maximum number of steps the solver can take. Keyword only
        argument.
    - `adjoint`: The adjoint method used to compute gradients through the fixed-point
        solve. Keyword only argument.
    - `throw`: How to report any failures. (E.g. an iterative solver running out of
        steps, or encountering divergent iterates.) If `True` then a failure will raise
        an error. If `False` then the returned solution object will have a `result`
        field indicating whether any failures occured. (See [`optimistix.Solution`][].)
        Keyword only argument.
    - `tags`: Lineax [tags](https://docs.kidger.site/lineax/api/tags/) describing
        any structure of the Hessian of `fn` with respect to `y`. Used with
        [`optimistix.ImplicitAdjoint`][] to implement the implicit function theorem as
        efficiently as possible. Keyword only argument.

    **Returns:**

    An [`optimistix.Solution`][] object.
    """

    y0 = jtu.tree_map(inexact_asarray, y0)
    if not has_aux:
        fn = NoneAux(fn)  # pyright: ignore
    fn = OutAsArray(fn)
    fn = eqx.filter_closure_convert(fn, y0, args)  # pyright: ignore
    fn = cast(Fn[Y, Scalar, Aux], fn)
    f_struct, aux_struct = fn.out_struct  # pyright: ignore[reportFunctionMemberAccess]
    if options is None:
        options = {}

    if not (
        isinstance(f_struct, jax.ShapeDtypeStruct)
        and f_struct.shape == ()
        and jnp.issubdtype(f_struct.dtype, jnp.floating)
    ):
        raise ValueError(
            "minimisation function must output a single floating-point scalar."
        )

    # Local import to avoid circular dependency: _root_find imports from _minimise.
    from ._root_find import AbstractRootFinder, root_find  # noqa: PLC0415

    if isinstance(solver, AbstractRootFinder):
        # Find roots of ∇f(y) = 0. The Jacobian of ∇f is the Hessian of f, so
        # passing `tags` through is semantically correct (both describe the same
        # matrix). This enables true second-order methods via JAX AD.
        sol = root_find(
            eqx.Partial(_to_grad_fn, fn),
            _ConcreteRootToMinimise(solver),  # pyright: ignore
            y0,
            args,
            options,
            has_aux=True,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw,
            tags=tags,
        )
        _, aux = sol.aux
        sol = eqx.tree_at(lambda s: s.aux, sol, aux)
        return sol

    return iterative_solve(
        fn,
        solver,
        y0,
        args,
        options,
        max_steps=max_steps,
        adjoint=adjoint,
        throw=throw,
        tags=tags,
        aux_struct=aux_struct,
        f_struct=f_struct,
        rewrite_fn=_rewrite_fn,
    )
