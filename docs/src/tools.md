# [Tools](@id tools-section)

## Functions evaluations

After calling one the [API](@ref) functions to get a function value, the
number of times that function was called is stored inside the
`NLPModel`. For instance

```@example
using ADNLPModels, LinearAlgebra, NLPModels
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
| [`neval_cons_lin`](@ref)        | Linear constraints |
| [`neval_cons_nln`](@ref)        | Nonlinear constraints |
| [`neval_jcon`](@ref)            | One constraint - unused |
| [`neval_jgrad`](@ref)           | Gradient of one constraints - unused |
| [`neval_jac`](@ref)             | Jacobian |
| [`neval_jac_lin`](@ref)         | Linear constraints Jacobian |
| [`neval_jac_nln`](@ref)         | Nonlinear constraints Jacobian |
| [`neval_jprod`](@ref)           | Product of Jacobian and vector |
| [`neval_jprod_lin`](@ref)       | Product of linear constraints Jacobian and vector |
| [`neval_jprod_nln`](@ref)       | Product of nonlinear constraints Jacobian and vector |
| [`neval_jtprod`](@ref)          | Product of transposed Jacobian and vector |
| [`neval_jtprod_lin`](@ref)      | Product of transposed linear constraints Jacobian and vector |
| [`neval_jtprod_nln`](@ref)      | Product of transposed nonlinear constraints Jacobian and vector |
| [`neval_hess`](@ref)            | Hessian |
| [`neval_hprod`](@ref)           | Product of Hessian and vector |
| [`neval_jhess`](@ref)           | Individual Lagrangian Hessian evaluations |
| [`neval_jhprod`](@ref)          | Product of Hessian of j-th function and vector |
| [`neval_residual`](@ref)        | Residual function of nonlinear least squares model |
| [`neval_jac_residual`](@ref)    | Jacobian of the residual |
| [`neval_jprod_residual`](@ref)  | Product of Jacobian of residual and vector |
| [`neval_jtprod_residual`](@ref) | Product of transposed Jacobian of residual and vector |
| [`neval_hess_residual`](@ref)   | Sum of Hessians of residuals |
| [`neval_jhess_residual`](@ref)  | Hessian of a residual component |
| [`neval_hprod_residual`](@ref)  | Product of Hessian of a residual component and vector |

To get the sum of all counters except `cons`, `jac`, `jprod` and `jtprod` called for a problem, use
[`sum_counters`](@ref).

```@example
using ADNLPModels, LinearAlgebra, NLPModels
nlp = ADNLPModel(x -> dot(x, x), zeros(2))
obj(nlp, rand(2))
grad(nlp, rand(2))
sum_counters(nlp)
```

## Querying problem type

There are some variable for querying the problem type:

- [`has_bounds`](@ref): True when not all variables are free.
- [`bound_constrained`](@ref): True for problems with bounded variables
  and no other constraints.
- [`equality_constrained`](@ref): True when problem is constrained only
  by equalities.
- [`has_equalities`](@ref): True when problem has at least one equality constraint.
- [`inequality_constrained`](@ref): True when problem is constrained
  by inequalities.
- [`has_inequalities`](@ref): True when problem has at least one inequality constraint that isn't a bound.
- [`linearly_constrained`](@ref): True when problem is constrained by
  equalities or inequalities known to be linear.
- [`unconstrained`](@ref): True when problem is not constrained.
