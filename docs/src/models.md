# Models

The following general models are implemented in this package:
- [ADNLPModel](@ref)
- [Derived Models](@ref)
  - [SlackModel](@ref)
  - [LBFGSModel](@ref)
  - [LSR1Model](@ref)

In addition, the following nonlinear least squares models are
implemented in this package:
- [ADNLSModel](@ref)
- [FeasibilityResidual](@ref)
- [LLSModel](@ref)
- [SlackNLSModel](@ref)
- [FeasibilityFormNLS](@ref)

There are other external models implemented. In particular,
- [AmplModel](https://github.com/JuliaSmoothOptimizers/AmplNLReader.jl)
- [CUTEstModel](https://github.com/JuliaSmoothOptimizers/CUTEst.jl)
- [MathProgNLPModel and MathProgNLSModel](https://github.com/JuliaSmoothOptimizers/NLPModelsJuMP.jl)
  using `JuMP/MPB`.

There are currently two models implemented in this package, besides the
external ones.

## ADNLPModel

```@docs
NLPModels.ADNLPModel
```

### Example

```@example
using NLPModels
f(x) = sum(x.^4)
x = [1.0; 0.5; 0.25; 0.125]
nlp = ADNLPModel(f, x)
grad(nlp, x)
```

## Derived Models

The following models are created from any given model, making some
modification to that model.

### SlackModel

```@docs
NLPModels.SlackModel
```

### Example

```@example
using NLPModels
f(x) = x[1]^2 + 4x[2]^2
c(x) = [x[1]*x[2] - 1]
x = [2.0; 2.0]
nlp = ADNLPModel(f, x, c=c, lcon=[0.0])
nlp_slack = SlackModel(nlp)
nlp_slack.meta.lvar
```

### LBFGSModel

```@docs
NLPModels.LBFGSModel
```

### LSR1Model

```@docs
NLPModels.LSR1Model
```

## ADNLSModel

```@docs
NLPModels.ADNLSModel
```

```@example
using NLPModels
F(x) = [x[1] - 1; 10*(x[2] - x[1]^2)]
nlp = ADNLSModel(F, [-1.2; 1.0], 2)
residual(nlp, nlp.meta.x0)
```

## FeasibilityResidual

```@docs
NLPModels.FeasibilityResidual
```

## LLSModel

```@docs
NLPModels.LLSModel
```

## SlackNLSModel

```@docs
NLPModels.SlackNLSModel
```

## FeasibilityFormNLS

```@docs
NLPModels.FeasibilityFormNLS
```
