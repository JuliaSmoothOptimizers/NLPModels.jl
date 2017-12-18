# API

As stated in the [home](home) page, we consider the nonlinear optimization
problem in the following format:
```math
\begin{align*}
\min \quad & f(x) \\
& c_L \leq c(x) \leq c_U \\
& \ell \leq x \leq u.
\end{align*}
```
To develop an optimization algorithm, we are usually worried not only with
``f(x)`` and ``c(x)``, but also with their derivatives.
Namely,

- ``\nabla f(x)``, the gradient of ``f`` at the point ``x``;
- ``\nabla^2 f(x)``, the Hessian of ``f`` at the point ``x``;
- ``J(x) = \nabla c(x)``, the Jacobian of ``c`` at the point ``x``;
- ``\nabla^2 f(x) + \sum_{i=1}^m \lambda_i \nabla^2 c_i(x)``,
  the Hessian of the Lagrangian function at the point ``(x,\lambda)``.

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
|-------------------|-------------------------------------------|
| ``f(x)``            | [obj](api.html#NLPModels.obj), [objgrad](api.html#NLPModels.objgrad), [objgrad!](api.html#NLPModels.objgrad!), [objcons](api.html#NLPModels.objcons), [objcons!](api.html#NLPModels.objcons!) |
| ``\nabla f(x)``     | [grad](api.html#NLPModels.grad), [grad!](api.html#NLPModels.grad!), [objgrad](api.html#NLPModels.objgrad), [objgrad!](api.html#NLPModels.objgrad!) |
| ``\nabla^2 f(x)``   | [hess](api.html#NLPModels.hess), [hess_op](api.html#NLPModels.hess_op), [hess_op!](api.html#NLPModels.hess_op!), [hess_coord](api.html#NLPModels.hess_coord), [hprod](api.html#NLPModels.hprod), [hprod!](api.html#NLPModels.hprod!) |
| ``c(x)``            | [cons](api.html#NLPModels.cons), [cons!](api.html#NLPModels.cons!), [objcons](api.html#NLPModels.objcons), [objcons!](api.html#NLPModels.objcons!) |
| ``J(x)``            | [jac](api.html#NLPModels.jac), [jac_op](api.html#NLPModels.jac_op), [jac_op!](api.html#NLPModels.jac_op!), [jac_coord](api.html#NLPModels.jac_coord), [jprod](api.html#NLPModels.jprod), [jprod!](api.html#NLPModels.jprod!), [jtprod](api.html#NLPModels.jtprod), [jtprod!](api.html#NLPModels.jtprod!) |
| ``\nabla^2 L(x,y)`` | [hess](api.html#NLPModels.hess), [hess_op](api.html#NLPModels.hess_op), [hess_coord](api.html#NLPModels.hess_coord), [hprod](api.html#NLPModels.hprod), [hprod!](api.html#NLPModels.hprod!) |

## API for NLSModels

For the Nonlinear Least Squares models, ``f(x) = \Vert F(x)\Vert^2``,
and these models have additional function to access the residual value
and its derivatives. Namely,

- ``J_F(x) = \nabla F(x)``
- ``\nabla^2 F_i(x)``

| Function            | function |
|---------------------|---|
| ``F(x)``            | [residual](api.html#NLPModels.residual), [residual!](api.html#NLPModels.residual!) |
| ``J_F(x)``          | [jac_residual](api.html#NLPModels.jac_residual), [jprod_residual](api.html#NLPModels.jprod_residual), [jprod_residual!](api.html#NLPModels.jprod_residual!), [jtprod_residual](api.html#NLPModels.jtprod_residual), [jtprod_residual!](api.html#NLPModels.jtprod_residual!), [jac_op_residual](api.html#NLPModels.jac_op_residual), [jac_op_residual!](api.html#NLPModels.jac_op_residual!) |
| ``\nabla^2 F_i(x)`` | [hess_residual](api.html#NLPModels.hess_residual), [hprod_residual](api.html#NLPModels.hprod_residual), [hprod_residual!](api.html#NLPModels.hprod_residual!), [hess_op_residual](api.html#NLPModels.hess_op_residual), [hess_op_residual!](api.html#NLPModels.hess_op_residual!) |


## AbstractNLPModel functions

```@docs
obj
grad
grad!
objgrad
objgrad!
cons
cons!
objcons
objcons!
jac_coord
jac
jac_op
jac_op!
jprod
jprod!
jtprod
jtprod!
hess_coord
hess
hess_op
hess_op!
hprod
hprod!
NLPtoMPB
reset!
```

## AbstractNLSModel

```@docs
residual
residual!
jac_residual
jprod_residual
jprod_residual!
jtprod_residual
jtprod_residual!
jac_op_residual
jac_op_residual!
hess_residual
hprod_residual
hprod_residual!
hess_op_residual
hess_op_residual!
```

## Derivative check

```@docs
gradient_check
jacobian_check
hessian_check
hessian_check_from_grad
```
