# Descents

??? abstract "`optimistix.AbstractDescent`"

    ::: optimistix.AbstractDescent
        options:
            members:
                - __call__

::: optimistix.SteepestDescent
    options:
        members:
            - __init__

---

::: optimistix.NonlinearCGDescent
    options:
        members:
            - __init__

---

::: optimistix.NewtonDescent
    options:
        members:
            - __init__

---

::: optimistix.DampedNewtonDescent
    options:
        members:
            - __init__

---

::: optimistix.IndirectDampedNewtonDescent
    options:
        members:
            - __init__

---

::: optimistix.DoglegDescent
    options:
        members:
            - __init__

---

::: optimistix.SteihaugCGDescent
    options:
        members:
            - __init__

---

# Linear solvers for Newton-type descents

[`optimistix.NewtonDescent`][], [`optimistix.IndirectDampedNewtonDescent`][], and related descents accept a `linear_solver` argument. The following solver is provided by Optimistix; any [`lineax.AbstractLinearSolver`][] may also be used.

::: optimistix.TruncatedCG
    options:
        members:
            - __init__
