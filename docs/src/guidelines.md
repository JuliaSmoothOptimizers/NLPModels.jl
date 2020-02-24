# [Guidelines for creating models](@id Guidelines)

These are guidelines for the creation of models using NLPModels to help keeping the models uniform, and for future reference in the creation of solvers.

## Bare minimum

Your model should derive from `AbstractNLPModel` or some other abstract class derived from it.
It is mandatory that it has a `meta :: NLPModelMeta` field, storing all the relevant problem information.
The minimum information that should be set is

- `nvar`: Number of variables
- `x0`: Starting point (defaults to `zeros`)
- `lvar`, `uvar`: Bounds on the variables (default to `(-∞,∞)`)
- `ncon`: Number of constraints (defaults to `0`)
- `lcon`, `ucon`: Bounds on the constraints (default to `(-∞,∞)`)
- `nnzh`: The length of the vectors used to store a triangle of the Hessian matrix in triplet format (defaults to `nvar * (nvar + 1) / 2`
- `nnzj`: The length of the vectors used to store the Jacobian matrix in triplet format (default to `nvar * ncon`)

The model also need to provide `Counters` information, the easiest way being defining `counters :: Counters`.
For instance:
```julia
mutable struct MyModel <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
end
```

If a model does not implement `counters`, then it needs to define `neval_xxx` functions, such as in the following macro:
```julia
for counter in fieldnames(Counters)
  @eval begin
    $counter(nlp :: MyModel) = SOMETHING
  end
end
```
One example of such model is the `SlackModel`, which stores an internal `model :: AbstractNLPModel`, thus defining
```julia
$counter(nlp :: SlackModel) = $counter(nlp.model)
```

There are about 30 functions in the NLPModels API, and a few with more than one signature, so it can be scary to implement them all.
Luckily, there are some functions with default definition to use other functions (e.g., `grad` is defined through `grad!` by default).
We compile here the list of functions that should be implemented for a complete API.

Here, the following notation apply:
- `nlp` is your model `MyModel <: AbstractNLPModel`
- `x` is the point where the function is evaluated
- `y` is the vector of Lagrange multipliers
- `g` is the gradient vector
- `H` is the Hessian matrix of the objective or Lagrangian
- `hrows`, `hcols`, and `hvals` are vectors storing the triplet form of the Hessian
- `c` is the constraints vector
- `J` is the Jacobian matrix of the constraints
- `jrows`, `jcols`, and `jvals` are vectors storing the triplet form of the Jacobian
- `v` is a vector of appropriate dimensions, generally used for matrix-vector products
- `Jv`, `Jtv`, `Hv` are vectors of appropriate dimensions, storing the result of matrix-vector products

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

## Testing your model

To test your model, in addition to writing specific testing functions, it is also advised to write consistency checks.
We provide 6 problems in our `test/problems` folder implemented both as `ADNLPModel` and by explicitly defining these problem as models.
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

## Expected behaviour

The following is a non-exhaustive list of expected behaviour for methods.

- All in place methods should also return the modified vectors.
- Vector inputs should have the correct size. If necessary, pass them using views or slices.
