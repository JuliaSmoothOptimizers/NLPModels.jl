# [Tools](@id tools-section)

## Functions evaluations

After calling one the [API](@ref) functions to get a function value, the
number of times that function was called is stored inside the
`NLPModel`. For instance

```@example
using NLPModels, LinearAlgebra
nlp = ADNLPModel(x -> dot(x, x), zeros(2))
for i = 1:100
    obj(nlp, rand(2))
end
neval_obj(nlp)
```

Some counters are available for all models, some are specific. In
particular, there are additional specific counters for the nonlinear
least squares models.

| Counter | Description |
|---|---|
| [`neval_obj`](@ref)             | Objective |
| [`neval_grad`](@ref)            | Gradient |
| [`neval_cons`](@ref)            | Constraints |
| [`neval_jcon`](@ref)            | One constraint - unused |
| [`neval_jgrad`](@ref)           | Gradient of one constraints - unused |
| [`neval_jac`](@ref)             | Jacobian |
| [`neval_jprod`](@ref)           | Product of Jacobian and vector |
| [`neval_jtprod`](@ref)          | Product of transposed Jacobian and vector |
| [`neval_hess`](@ref)            | Hessian |
| [`neval_hprod`](@ref)           | Product of Hessian and vector |
| [`neval_jhprod`](@ref)          | Product of Hessian of j-th function and vector |
| [`neval_residual`](@ref)        | Residual function of nonlinear least squares model |
| [`neval_jac_residual`](@ref)    | Jacobian of the residual |
| [`neval_jprod_residual`](@ref)  | Product of Jacobian of residual and vector |
| [`neval_jtprod_residual`](@ref) | Product of transposed Jacobian of residual and vector |
| [`neval_hess_residual`](@ref)   | Sum of Hessians of residuals |
| [`neval_jhess_residual`](@ref)  | Hessian of a residual component |
| [`neval_hprod_residual`](@ref)  | Product of Hessian of a residual component and vector |

To get the sum of all counters called for a problem, use
[`sum_counters`](@ref).

```@example
using NLPModels, LinearAlgebra
nlp = ADNLPModel(x -> dot(x, x), zeros(2))
obj(nlp, rand(2))
grad(nlp, rand(2))
sum_counters(nlp)
```

## Querying problem type

There are some variable for querying the problem type:

- [`bound_constrained`](@ref): True for problems with bounded variables
  and no other constraints.
- [`equality_constrained`](@ref): True when problem is constrained only
  by equalities.
- [`has_bounds`](@ref): True when not all variables are free.
- [`inequality_constrained`](@ref): True when problem is constrained
  by inequalities.
- [`linearly_constrained`](@ref): True when problem is constrained by
  equalities or inequalities known to be linear.
- [`unconstrained`](@ref): True when problem is not constrained.

## Docs

```@docs
neval_obj
neval_grad
neval_cons
neval_jcon
neval_jgrad
neval_jac
neval_jprod
neval_jtprod
neval_hess
neval_hprod
neval_jhprod
neval_residual
neval_jac_residual
neval_jprod_residual
neval_jtprod_residual
neval_hess_residual
neval_jhess_residual
neval_hprod_residual
sum_counters
bound_constrained
equality_constrained
has_bounds
inequality_constrained
linearly_constrained
unconstrained
```
