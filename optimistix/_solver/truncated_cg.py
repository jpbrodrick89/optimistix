"""Truncated CG linear solver for indefinite operators (Newton-CG)."""

from typing import Any, TypeAlias, cast

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from lineax._solution import RESULTS as _lxRESULTS

from .._misc import tree_dot, tree_full_like, tree_where


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
    - `"delta"`: Trust-region radius. If provided, the solver also exits early when
        `‖p‖ ≥ delta`, returning the current iterate before the boundary-crossing
        step. The search direction at the exit point is returned in
        `stats["direction"]`, allowing the caller to project onto the trust-region
        boundary if needed. Defaults to `jnp.inf` (no trust-region constraint).
    """

    rtol: float
    max_steps: int | None = None

    def __check_init__(self):
        if isinstance(self.rtol, (int, float)) and self.rtol < 0:
            raise ValueError("`TruncatedCG` requires `rtol >= 0`.")
        if (
            isinstance(self.rtol, (int, float))
            and self.rtol == 0
            and self.max_steps is None
        ):
            raise ValueError(
                "Must specify `rtol > 0` or `max_steps` (or both) for `TruncatedCG`."
            )

    def init(
        self, operator: lx.AbstractLinearOperator, options: dict[str, Any]
    ) -> _TruncatedCGState:
        del options
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
        delta = options.get("delta", jnp.inf)
        delta_sq = delta**2
        rtol = options.get("rtol", self.rtol)

        # Standard CG initialisation for solving H p = vector:
        #   r0 = vector - H*0 = vector  (initial residual)
        #   d0 = r0                     (initial search direction)
        r0 = vector
        r0_norm_sq = cast(Array, tree_dot(r0, r0))
        tol_sq = rtol**2 * r0_norm_sq
        p0 = tree_full_like(r0, 0)

        class _CGState(eqx.Module):
            p: Any
            r: Any
            d: Any
            rr: Scalar
            result_p: Any  # frozen at first exit
            result_d: Any  # frozen at first exit (search direction, for boundary projection)
            done: Bool[Array, ""]
            negative_curvature: Bool[Array, ""]

        def body_fn(_, cg_state: _CGState) -> _CGState:
            Hd = operator.mv(cg_state.d)
            dHd = cast(Array, tree_dot(cg_state.d, Hd))

            # Exit 1: negative curvature — d^T H d <= 0.
            neg_curv = dHd <= jnp.finfo(dHd.dtype).eps

            safe_dHd = jnp.where(neg_curv, 1.0, dHd)
            alpha = cg_state.rr / safe_dHd

            p_new = (cg_state.p**ω + alpha * cg_state.d**ω).ω
            p_new_norm_sq = cast(Array, tree_dot(p_new, p_new))

            # Exit 2: trust-region boundary — ||p_new|| >= delta.
            past_boundary = p_new_norm_sq >= delta_sq

            # Standard CG residual update: r_{k+1} = r_k - alpha * H d_k.
            r_new = (cg_state.r**ω - alpha * Hd**ω).ω
            rr_new = cast(Array, tree_dot(r_new, r_new))

            # Exit 3: residual converged.
            converged = rr_new < tol_sq

            done_now = neg_curv | past_boundary | converged

            # Choose what to return as the solution at this exit point:
            #   neg_curv:      current p (before the bad step); if p=0 (first step),
            #                  fall back to d (= r0 = -g), the steepest-descent dir.
            #   boundary/conv: p_new (the step that triggered or the converged point).
            p_is_zero = cast(Array, tree_dot(cg_state.p, cg_state.p)) <= 0
            neg_curv_result = tree_where(p_is_zero, cg_state.d, cg_state.p)
            result_now = tree_where(neg_curv, neg_curv_result, p_new)

            safe_rr = jnp.where(
                cg_state.rr > jnp.finfo(cg_state.rr.dtype).eps, cg_state.rr, 1.0
            )
            beta = rr_new / safe_rr
            # Standard CG direction update: d_{k+1} = r_{k+1} + beta * d_k.
            d_new = (r_new**ω + beta * cg_state.d**ω).ω

            new_done = cg_state.done | done_now
            # Freeze result_p and result_d at the first exit; keep advancing otherwise.
            new_result_p = tree_where(
                cg_state.done,
                cg_state.result_p,
                tree_where(done_now, result_now, p_new),
            )
            new_result_d = tree_where(
                cg_state.done,
                cg_state.result_d,
                tree_where(done_now, cg_state.d, d_new),
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
                result_d=new_result_d,
                done=new_done,
                negative_curvature=cg_state.negative_curvature | neg_curv,
            )

        if self.max_steps is None:
            leaves, _ = jtu.tree_flatten(vector)
            max_steps = 20 * sum(leaf.size for leaf in leaves)
        else:
            max_steps = self.max_steps

        # Initialise result_p = r0 so that if max_steps=0, or if negative curvature
        # fires on the very first step (p=0), we return r0 = -g (steepest descent).
        init_cg = _CGState(
            p=p0,
            r=r0,
            d=r0,
            rr=r0_norm_sq,
            result_p=r0,
            result_d=r0,
            done=jnp.array(False),
            negative_curvature=jnp.array(False),
        )

        final = lax.fori_loop(0, max_steps, body_fn, init_cg)

        result = _lxRESULTS.where(final.done, _lxRESULTS.successful, _lxRESULTS.max_steps_reached)
        stats = {
            "num_steps": jnp.array(max_steps),
            "negative_curvature": final.negative_curvature,
            "direction": final.result_d,
        }
        return final.result_p, result, stats

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

- `rtol`: Relative tolerance for CG convergence. CG exits when
    `‖r_k‖ < rtol * ‖r_0‖`. Can be overridden per-call via `options["rtol"]`.
- `max_steps`: Maximum number of CG iterations. Defaults to `20 * n` where `n`
    is the size of the problem (matching scipy's Newton-CG default). If
    `max_steps` is reached without convergence the solve returns
    `_lxRESULTS.max_steps_reached`.
"""
