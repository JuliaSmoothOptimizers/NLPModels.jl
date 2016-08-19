# NLPModels.jl documentation

This package provides general guidelines to represent optimization problems in
Julia and a standardized API to evaluate the functions and their derivatives.
The main objective is to be able to rely on that API when designing optimization
solvers in Julia.

Optimization problems are represented by an instance/subtype of `AbstractNLPModel`.
Such instances are composed of

- an instance of `NLPModelMeta`, which provides information about the problem,
  including the number of variables, constraints, bounds on the variables, etc.
- other data specific to the provenance of the problem.

## Internal Interfaces

 - [`SimpleNLPModel`](@ref): Uses
   [`ForwardDiff`](http://github.com/JuliaDiff/ForwardDiff.jl) to compute the
   derivatives. It has a very simple interface.
 - [`JuMPNLPModel`](@ref): Uses a [`JuMP`](https://github.com/JuliaOpt/JuMP.jl) model.
  - [`SlackModel`](@ref): Creates an equality constrained problem with bounds
    on the variables using an existing NLPModel.

## External Interfaces

 - `AmplModel`: Defined in
   [`AmplNLReader.jl`](https://github.com/JuliaSmoothOptimizers/AmplNLReader.jl)
   for problems modeled using [AMPL](http://www.ampl.com)
 - `CUTEstModel`: Defined in
   [`CUTEst.jl`](https://github.com/JuliaSmoothOptimizers/CUTEst.jl) for
   problems from [CUTEst](https://ccpforge.cse.rl.ac.uk/gf/project/cutest/wiki).

If you want your interface here, open a PR.
