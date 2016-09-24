# API

As stated in the [home](home) page, we consider the nonlinear optimization
problem in the following format:
\begin{align*}
\min \quad & f(x) \\\\
& c_L \leq c(x) \leq c_U \\\\
& \ell \leq x \leq u.
\end{align*}
To develop an optimization algorithm, we are usually worried not only with
$f(x)$ and $c(x)$, but also with their derivatives.
Namely,

- $\nabla f(x)$, the gradient of $f$ at the point $x$;
- $\nabla^2 f(x)$, the Hessian of $f$ at the point $x$;
- $J(x) = \nabla c(x)$, the Jacobian of $c$ at the point $x$;
- $\nabla^2 f(x) + \sum_{i=1}^m \lambda_i \nabla^2 c_i(x)$,
  the Hessian of the Lagrangian function at the point $(x,\lambda)$.

There are many ways to access some of these values, so here is a little
reference guide.

## Reference guide

The following naming should be easy enough to follow.
If not, click on the link and go to the description.

- `!` means inplace;
- `_coord` means coordinate format;
- `prod` means matrix-vector product;
- `_op` means operator (as in [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl)).

Feel free to open an issue to suggest other methods that should apply to all
NLPModels instances.

| Function          | NLPModels function                                                                                                                                                                                   |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| $f(x)$            | [obj](api/#NLPModels.obj)                                                                                                                                                                            |
| $\nabla f(x)$     | [grad](api/#NLPModels.grad), [grad!](api/#NLPModels.grad!)                                                                                                                                           |
| $\nabla^2 f(x)$   | [hess](api/#NLPModels.hess), [hess_op](api/#NLPModels.hess_op), [hess_coord](api/#NLPModels.hess_coord), [hprod](api/#NLPModels.hprod), [hprod!](api/#NLPModels.hprod!)                              |
| $c(x)$            | [cons](api/#NLPModels.cons), [cons!](api/#NLPModels.cons!)                                                                                                                                           |
| $J(x)$            | [jac](api/#NLPModels.jac), [jac_coord](api/#NLPModels.jac_coord), [jprod](api/#NLPModels.jprod), [jprod!](api/#NLPModels.jprod!), [jtprod](api/#NLPModels.jtprod), [jtprod!](api/#NLPModels.jtprod!) |
| $\nabla^2 L(x,y)$ | [hess](api/#NLPModels.hess), [hess_op](api/#NLPModels.hess_op), [hess_coord](api/#NLPModels.hess_coord), [hprod](api/#NLPModels.hprod), [hprod!](api/#NLPModels.hprod!)                              |


## AbstractNLPModel functions

```@docs
obj
grad
grad!
cons
cons!
jac_coord
jac
jprod
jprod!
jtprod
jtprod!
hess_coord
hess
hess_op
hprod
hprod!
NLPtoMPB
reset!
```

## Derivative check

```@docs
gradient_check
jacobian_check
hessian_check
hessian_check_from_grad
```
