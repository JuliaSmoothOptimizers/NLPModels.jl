# Models

The following is a list of packages implementing the NLPModels API.

If you want your package listed here, open a Pull Request.

If you want to create your own interface, check these [Guidelines](@ref).
## Packages

- [NLPModelsModifiers.jl](https://github.com/JuliaSmoothOptimizers/NLPModelsModifiers.jl):
  Models that modify existing models.
  For instance, creating slack variables, or moving constraints into the objective functions, or using Quasi-Newton LBFSG approximations to the Hessian.
- [ADNLPModels.jl](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl):
  Models with automatic differentiation.
- [CUTEst.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl):
  For problems from [CUTEst](https://github.com/ralna/CUTEst/wiki).
- [AmplNLReader.jl](https://github.com/JuliaSmoothOptimizers/AmplNLReader.jl):
  For problems modeled using [AMPL](https://ampl.com)
- [NLPModelsJuMP.jl](https://github.com/JuliaSmoothOptimizers/NLPModelsJuMP.jl):
  For problems modeled using [JuMP.jl](https://github.com/jump-dev/JuMP.jl).
- [QuadraticModels.jl](https://github.com/JuliaSmoothOptimizers/QuadraticModels.jl):
  For problems with linear constraints and a quadratic objective (LCQP).
- [LLSModels.jl](https://github.com/JuliaSmoothOptimizers/LLSModels.jl):
  Creates a linear least squares model.
- [PDENLPModels.jl](https://github.com/JuliaSmoothOptimizers/PDENLPModels.jl):
  For PDE-constrained problems.
- [BlockNLPModels.jl](https://github.com/exanauts/BlockNLPModels.jl): For modeling block structured nonlinear optimization problems.

## Model internals

- [`AbstractNLPModel`](@ref)
- [`AbstractNLSModel`](@ref)
- [`AbstractNLPModelMeta`](@ref)
- [`NLPModelMeta`](@ref)
- [`NLSMeta`](@ref)
- [`nls_meta`](@ref)
