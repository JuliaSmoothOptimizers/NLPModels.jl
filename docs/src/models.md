# Models

The following general models are implemented in this package:
- [ADNLPModel](#ADNLPModel-1)
- [MathProgNLPModel](#MathProgNLPModel-1)
- [SimpleNLPModel](#SimpleNLPModel-1)
- [Derived Models](#Derived-Models-1)
  - [SlackModel](#SlackModel-1)
  - [LBFGSModel](#LBFGSModel-1)
  - [LSR1Model](#LSR1Model-1)

In addition, the following nonlinear least squares models are
implemented in this package:
- [ADNLSModel](#ADNLSModel-1)
- [FeasibilityResidual](#FeasibilityResidual-1)
- [LLSModel](#LLSModel-1)
- [SimpleNLSModel](#SimpleNLSModel-1)

There are other external models implemented. In particular,
- [AmplModel](https://github.com/JuliaSmoothOptimizers/AmplNLReader.jl)
- [CUTEstModel](https://github.com/JuliaSmoothOptimizers/CUTEstModel.jl)

There are currently three models implemented in this package, besides the
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

## MathProgNLPModel

```@docs
NLPModels.MathProgNLPModel
```

### Example

```@example
using NLPModels, MathProgBase, JuMP
m = Model()
@variable(m, x[1:4])
@NLobjective(m, Min, sum(x[i]^4 for i=1:4))
nlp = MathProgNLPModel(m)
x0 = [1.0; 0.5; 0.25; 0.125]
grad(nlp, x0)
```

## SimpleNLPModel

```@docs
NLPModels.SimpleNLPModel
```

### Example

```@example
using NLPModels
f(x) = sum(x.^4)
g(x) = 4*x.^3
x = [1.0; 0.5; 0.25; 0.125]
nlp = SimpleNLPModel(f, x, g=g)
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

## SimpleNLSModel

```@docs
NLPModels.SimpleNLSModel
```
