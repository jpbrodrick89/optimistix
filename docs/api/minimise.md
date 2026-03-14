# Minimisation

In addition to the following, note that the [Optax](https://github.com/deepmind/optax) library offers an extensive collection of minimisers via first-order gradient methods -- as are in widespread use for neural networks. If you would like to use these through the Optimistix API then an [`optimistix.OptaxMinimiser`][] wrapper is provided.

::: optimistix.minimise

---

## Second-order minimisation via root finders

!!! info

    In addition to the [`optimistix.AbstractMinimiser`][] solvers listed below, any [root finder](./root_find.md) may also be used as the `solver`. Optimistix will rewrite the minimisation problem as a root-finding problem on the gradient, i.e. finding `y` such that `∇f(y) = 0`. This unlocks **true second-order methods**: the Jacobian of `∇f` is the Hessian of `f`, so solvers like [`optimistix.Newton`][] will internally compute exact Hessian-vector products via JAX's forward-over-reverse AD — with no need to explicitly form or store the Hessian matrix.

### Newton--Krylov (large-scale, Hessian-free)

For large problems where forming the full Hessian is too expensive, a Krylov iterative solver can be used as the linear solver inside Newton, computing only Hessian-vector products. This is known as a **Newton--Krylov** or **Hessian-free** method:

```python
import jax.numpy as jnp
import lineax as lx
import optimistix as optx

def rosenbrock(y, args):
    return (1 - y[0])**2 + 100*(y[1] - y[0]**2)**2

# Newton with GMRES as the inner linear solver — only Hessian-vector products
# are computed, never the full Hessian matrix.
solver = optx.Newton(
    rtol=1e-6,
    atol=1e-6,
    linear_solver=lx.GMRES(atol=1e-6, rtol=1e-6),
)
y0 = jnp.array([0.0, 0.0])
sol = optx.minimise(rosenbrock, solver, y0)
```

!!! note "MINRES"

    For this use case, [MINRES](https://web.stanford.edu/group/SOL/software/minres/) would be the theoretically preferred Krylov solver, as the Hessian is always symmetric and MINRES is designed for symmetric indefinite systems (using a short 3-term recurrence rather than GMRES's growing Arnoldi basis). MINRES is not yet available in Lineax, but is planned.

### Direct solver (small- to medium-scale)

For problems where `y` is small enough that the full Hessian can be assembled and factored, a direct linear solver gives the exact Newton step and typically converges in very few iterations:

```python
# Newton with the default direct solver — assembles and factors the full Hessian.
# Quadratic convergence near the minimum.
solver = optx.Newton(rtol=1e-8, atol=1e-8)  # uses lx.AutoLinearSolver by default
sol = optx.minimise(rosenbrock, solver, y0)
```

The direct and Krylov paths share identical outer-loop logic; only the `linear_solver` argument differs. When the Hessian is positive definite, a modified LDL^T factorisation (once available in Lineax) will provide the most robust direct-solver option.

---

[`optimistix.minimise`][] supports any of the following minimisers.

??? abstract "`optimistix.AbstractMinimiser`"

    ::: optimistix.AbstractMinimiser
        options:
            members:
                - init
                - step
                - terminate
                - postprocess

??? abstract "`optimistix.AbstractGradientDescent`"

    ::: optimistix.AbstractGradientDescent
        options:
            members: none

::: optimistix.GradientDescent
    options:
        members:
            - __init__

---

??? abstract "`optimistix.AbstractQuasiNewton`"

    ::: optimistix.AbstractQuasiNewton
        options:
            members:
                - init_hessian
                - update_hessian

??? abstract "`optimistix.AbstractBFGS`"

    ::: optimistix.AbstractBFGS
        options:
            members: none

::: optimistix.BFGS
    options:
        members:
            - __init__

??? abstract "`optimistix.AbstractDFP`"

    ::: optimistix.AbstractDFP
        options:
            members: none

::: optimistix.DFP
    options:
        members:
            - __init__

??? abstract "`optimistix.AbstractLBFGS`"

    ::: optimistix.AbstractLBFGS
        options:
            members: none

::: optimistix.LBFGS
    options:
        members:
            - __init__

---

::: optimistix.OptaxMinimiser
    options:
        members:
            - __init__

`optim` in [`optimistix.OptaxMinimiser`][] is an instance of an Optax minimiser. For example, correct usage is `optimistix.OptaxMinimiser(optax.adam(...), ...)`, not `optimistix.OptaxMinimiser(optax.adam, ...)`.

---

::: optimistix.NonlinearCG
    options:
        members:
            - __init__

[`optimistix.NonlinearCG`][] supports several different methods for computing its β parameter. If you are trying multiple solvers to see which works best on your problem, then you may wish to try all four versions of nonlinear CG. These can each be passed as `NonlinearCG(..., method=...)`.

::: optimistix.polak_ribiere

::: optimistix.fletcher_reeves

::: optimistix.hestenes_stiefel

::: optimistix.dai_yuan

---

::: optimistix.NelderMead
    options:
        members:
            - __init__

---

::: optimistix.GoldenSearch
    options:
        members:
            - __init__

::: optimistix.BestSoFarMinimiser
    options:
        members:
            - __init__
