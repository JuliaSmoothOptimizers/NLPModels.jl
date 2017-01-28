# Models

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

## SlackModel

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
