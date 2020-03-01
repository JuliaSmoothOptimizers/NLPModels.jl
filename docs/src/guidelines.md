# [Guidelines for creating models](@id Guidelines)

These are guidelines for the creation of models using NLPModels to help keeping the models uniform, and for future reference in the creation of solvers.

Table of contents:
- [Bare minimum](@ref bare-minimum)
- [Expected behaviour](@ref expected-behaviour)
- [Advanced counters](@ref advanced-counters)
- [Advanced tests](@ref advanced-tests)

## [Bare minimum](@id bare-minimum)

Your model should derive from `AbstractNLPModel` or some other abstract class derived from it.
It is mandatory that it have a `meta :: NLPModelMeta` field, storing all the relevant problem information.
The model also needs to provide `Counters` information. The easiest way is to define `counters :: Counters`.
For instance:
```julia
mutable struct MyModel <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
end
```
For alternatives to storing `Counters` in the model, check [advanced counters](@ref advanced-counters).
The minimum information that should be set for your model through `NLPModelMeta` is `nvar`, the number of variables.
The following is a valid constructor for `MyModel`:
```julia
function MyModel()
  return MyModel(NLPModelMeta(5), Counters())
end
```
More information can be passed to `NLPModelMeta`.
See the full list [here](https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/master/src/nlp_types.jl#L28).
The essential fields are
- `x0`: Starting point (defaults to `zeros`)
- `lvar`, `uvar`: Bounds on the variables (default to `(-∞,∞)`)
- `ncon`: Number of constraints (defaults to `0`)
- `lcon`, `ucon`: Bounds on the constraints (default to `(-∞,∞)`)
- `nnzh`: The length of the vectors used to store a triangle of the Hessian in triplet format (defaults to `nvar * (nvar + 1) / 2`
- `nnzj`: The length of the vectors used to store the Jacobian in triplet format (default to `nvar * ncon`)

There are about 30 functions in the NLPModels API, and a few with more than one signature.
Luckily, many have a default implementation.
We collect here the list of functions that should be implemented for a complete API.

Here, the following notation apply:
- `nlp` is your instance of `MyModel <: AbstractNLPModel`
- `x` is the point where the function is evaluated
- `y` is the vector of Lagrange multipliers (for constrained problems only)
- `g` is the gradient vector
- `H` is the Hessian of the objective or Lagrangian
- `hrows`, `hcols`, and `hvals` are vectors storing the triplet form of the Hessian
- `c` is the vector of constraints
- `J` is the Jacobian of the constraints
- `jrows`, `jcols`, and `jvals` are vectors storing the triplet form of the Jacobian
- `v` is a vector of appropriate dimensions, generally used for operator-vector products
- `Jv`, `Jtv`, `Hv` are vectors of appropriate dimensions, storing the result of operator-vector products

The following functions should be defined:

- Objective (unconstrained models only need to worry about these)
  - `obj(nlp, x)`
  - `grad!(nlp, x, g)`
  - `hess_structure!(nlp, hrows, hcols)`
  - `hess_coord!(nlp, x, hvals; obj_weight=1)`
  - `hprod!(nlp, x, v, Hv; obj_weight=1)` (actually defaults to calling the constrained case)
- Constraints (constrained models need to worry about these and the ones above)
  - `cons!(nlp, x, c)`
  - `jac_structure!(nlp, jrows, jcols)`
  - `jac_coord!(nlp, x, jvals)`
  - `jprod!(nlp, x, v, Jv)`
  - `jtprod!(nlp, x, v, Jtv)`
  - `hess_coord!(nlp, x, y, hvals; obj_weight=1)`
  - `hprod!(nlp, x, y, v, Hv; obj_weight=1)`

## [Expected behaviour](@id expected-behaviour)

The following is a non-exhaustive list of expected behaviour for methods.

- All in place methods should also return the modified vectors.
- Vector inputs should have the correct size. If necessary, the user should pass them using views or slices.
- The triplet format does not assume order nor uniqueness.

## [Advanced counters](@id advanced-counters)

If a model does not implement `counters`, then it needs to define
- `neval_xxx(nlp)` - get field `xxx` of `Counters`
- `reset!(nlp)` - resetting all counters
- `increment!(nlp, s)` - increment counter `s`
For instance
```julia
for counter in fieldnames(Counters)
  @eval begin
    $counter(nlp :: MyModel) = SOMETHING
  end
end
function reset!(nlp :: MyModel)
  RESET COUNTERS
end
function increment!(nlp :: MyModel, s :: Symbol)
  INCREMENT COUNTER s
end
```
One example of such model is the `SlackModel`, which stores an internal `model :: AbstractNLPModel`, thus defining
```julia
$counter(nlp :: SlackModel) = $counter(nlp.model)
reset!(nlp :: SlackModel) = reset!(nlp.model)
increment!(nlp :: SlackModel, s :: Symbol) = increment!(nlp.model, s)
```
This construction can be replicated calling the macro `@default_counters Model inner`.
In the case of SlackModel, the equivalent call is
```julia
@default_counters SlackModel model
```

## [Advanced tests](@id advanced-tests)

To test your model, in addition to writing specific test functions, it is also advised to write consistency checks.
If your model can implement general problems, you can use the 6 problems in our `test/problems` folder implemented both as `ADNLPModel` and by explicitly defining these problem as models.
These can be used to verify that the implementation of your model is correct through the `consistent_nlps` function.
The simplest way to use these would be something like
```julia
for problem in ["BROWNDEN", "HS5", "HS6", "HS10", "HS11", "HS14"]
  @printf("Checking problem %-20s", problem)
  nlp_ad = eval(Meta.parse(lowercase(problem) * "_autodiff"))() # e.g. hs5_autodiff()
  nlp_man = eval(Meta.parse(problem))() # e.g. HS5()
  nlp_your = ...
  nlps = [nlp_ad, nlp_man, nlp_your]
  consistent_nlps(nlps)
end
```

Models with specific purposes can make use of the consistency checks by defining equivalent problems with `ADNLPModel` and testing them.
For instance, the following model is a regularization model defined by an existing model `inner`, a regularization parameter `ρ`, and a fixed point `z`:
```julia
mutable struct RegNLP <: AbstractNLPModel
  meta :: NLPModelMeta
  inner :: AbstractNLPModel
  ρ
  z
end
```
Assuming that all unconstrained functions are defined, the following tests will make sure that `RegNLP` is consistent with a specific `ADNLPModel`.
```julia
include(joinpath(dirname(pathof(NLPModels)), "..", "test", "consistency.jl"))

f(x) = (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2
nlp = ADNLPModel(f, [-1.2; 1.0])
ρ = rand()
z = rand(2)
rnlp = RegNLP(nlp, ρ, z)
manual = ADNLPModel(x -> f(x) + ρ * norm(x - z)^2 / 2, [-1.2; 1.0])

consistent_nlps([rnlp, manual])
```
The complete example is available in the repository [RegularizationModel.jl](https://github.com/JuliaSmoothOptimizers/RegularizationModel.jl).

