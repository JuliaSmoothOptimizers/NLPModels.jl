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
mutable struct MyModel{T, S} <: AbstractNLPModel{T, S}
  meta :: NLPModelMeta{T, S}
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
See the full list [here](https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/main/src/nlp/meta.jl#L32).
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

Here, the following notation applies:
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
  - `hess_structure!(nlp, hrows, hcols)` (sparse Hessian)
  - `hess_coord!(nlp, x, hvals; obj_weight=1)` (sparse Hessian)
  - `hess_dense!(nlp, x, Hx; obj_weight=1)` (dense Hessian)
  - `hprod!(nlp, x, v, Hv; obj_weight=1)` (actually defaults to calling the constrained case)
- Constraints (constrained models need to worry about these and the ones above)
  - `cons_lin!(nlp, x, c)`
  - `cons_nln!(nlp, x, c)`
  - `jac_lin_structure!(nlp, jrows, jcols)` (sparse Jacobian)
  - `jac_nln_structure!(nlp, jrows, jcols)` (sparse Jacobian)
  - `jac_lin_coord!(nlp, x, jvals)` (sparse Jacobian)
  - `jac_nln_coord!(nlp, x, jvals)` (sparse Jacobian)
  - `jac_dense!(nlp, x, Jx)` (dense Jacobian)
  - `jprod_lin!(nlp, x, v, Jv)`
  - `jprod_nln!(nlp, x, v, Jv)`
  - `jtprod_lin!(nlp, x, v, Jtv)`
  - `jtprod_nln!(nlp, x, v, Jtv)`
  - `hess_coord!(nlp, x, y, hvals; obj_weight=1)` (sparse Hessian)
  - `hess_dense!(nlp, x, y, Hx; obj_weight=1)` (dense Hessian)
  - `hprod!(nlp, x, y, v, Hv; obj_weight=1)`

The linear constraints are specified at the initialization of the `NLPModelMeta` using the keyword arguement `lin`.
The indices of linear and nonlinear constraints are respectively available in `nlp.meta.lin` and `nlp.meta.nln`.
If your model uses only linear (resp. nonlinear) constraints, then it suffices to implement the `*_lin` (resp. `*_nln`) functions.
Alternatively, one could implement only the functions without the suffixes `_nln!` (e.g., only `cons!`), but this might run into errors with tools differentiating linear and nonlinear constraints.

If the Jacobian or the Hessian of the Lagrangian is dense, there is no need to implement the corresponding `*_structure!` and `*_coord!` methods.
Only the corresponding `*_dense!` methods need to be implemented.
This is specified at the initialization of [`NLPModelMeta`](@ref) through the keyword arguments `sparse_jacobian` and `sparse_hessian`.

## [Availability of the API](@id availability-api)

If only a subset of the functions listed above is implemented, you can indicate which ones are not available when creating the [`NLPModelMeta`](@ref), using the keyword arguments
`grad_available`, `jac_available`, `hess_available`, `jprod_available`, `jtprod_available`, and `hprod_available`.

By default, `grad_available`, `hess_available`, and `hprod_available` are set to `true`.
For constrained problems (`ncon > 0`), the fields `jac_available`, `jprod_available`, and `jtprod_available` are also set to `true`.
For unconstrained problems (`ncon == 0`), they default to `false`.

## [Expected behaviour](@id expected-behaviour)

The following is a non-exhaustive list of expected behaviour for methods.

- All in place methods should also return the modified vectors.
- Vector inputs should have the correct size. If necessary, the user should pass them using views or slices.
- The triplet format does not assume order nor uniqueness.

## [Show](@id show)

To further specialize your model, you can also define `show_header` and possibly `show`.
The default `show_header` simply prints the `typeof` the NLPModel, so it should be specialized with the specific information that you prefer. For instance, `SlackModel` defines
```julia
show_header(io :: IO, nlp :: SlackModel) = println(io, "SlackModel - Model with slack variables")
```
Furthermore, we define a general `show` that calls `show_header` and specific `show` functions for the `meta` and the `counters`. If your model does not have `counters` in the default location, you must define `show` for them as well. Alternatively, you may desire to change the behaviour of show. Here is an example, again from `SlackModel`:
```julia
function show(io :: IO, nlp :: SlackModel)
  show_header(io, nlp)
  show(io, nlp.meta)
  show(io, nlp.model.counters)
end
```

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

Furthermore, the `show` method has to be updated with the correct direction of `counter`. See [show](@ref show) for more information.

## [Advanced tests](@id advanced-tests)

We have created the package [NLPModelsTest.jl](https://github.com/JuliaSmoothOptimizers/NLPModelsTest.jl) which defines test functions and problems.
To make sure that your model is robust, we recommend using it in the test suite of your package.
