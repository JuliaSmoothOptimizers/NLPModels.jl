# API

As stated in the [Home](@ref) page, we consider the nonlinear optimization
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
| ``f(x)``            | [`obj`](@ref), [`objgrad`](@ref), [`objgrad!`](@ref), [`objcons`](@ref), [`objcons!`](@ref) |
| ``\nabla f(x)``     | [`grad`](@ref), [`grad!`](@ref), [`objgrad`](@ref), [`objgrad!`](@ref) |
| ``\nabla^2 f(x)``   | [`hess`](@ref), [`hess_op`](@ref), [`hess_op!`](@ref), [`hess_coord`](@ref), [`hess_coord`](@ref), [`hess_structure`](@ref), [`hess_structure!`](@ref), [`hprod`](@ref), [`hprod!`](@ref) |
| ``c(x)``            | [`cons`](@ref), [`cons!`](@ref), [`objcons`](@ref), [`objcons!`](@ref) |
| ``J(x)``            | [`jac`](@ref), [`jac_op`](@ref), [`jac_op!`](@ref), [`jac_coord`](@ref), [`jac_coord!`](@ref), [`jac_structure`](@ref), [`jprod`](@ref), [`jprod!`](@ref), [`jtprod`](@ref), [`jtprod!`](@ref) |
| ``\nabla^2 L(x,y)`` | [`hess`](@ref), [`hess_op`](@ref), [`hess_coord`](@ref), [`hess_coord!`](@ref), [`hess_structure`](@ref), [`hess_structure!`](@ref), [`hprod`](@ref), [`hprod!`](@ref) |

## API for NLSModels

For the Nonlinear Least Squares models, ``f(x) = \Vert F(x)\Vert^2``,
and these models have additional function to access the residual value
and its derivatives. Namely,

- ``J_F(x) = \nabla F(x)``
- ``\nabla^2 F_i(x)``

| Function            | function |
|---------------------|---|
| ``F(x)``            | [`residual`](@ref), [`residual!`](@ref) |
| ``J_F(x)``          | [`jac_residual`](@ref), [`jac_coord_residual`](@ref), [`jac_coord_residual!`](@ref), [`jac_structure_residual`](@ref), [`jprod_residual`](@ref), [`jprod_residual!`](@ref), [`jtprod_residual`](@ref), [`jtprod_residual!`](@ref), [`jac_op_residual`](@ref), [`jac_op_residual!`](@ref) |
| ``\nabla^2 F_i(x)`` | [`hess_residual`](@ref), [`hess_coord_residual`](@ref), [`hess_coord_residual!`](@ref), [`hess_structure_residual`](@ref), [`hess_structure_residual!`](@ref), [`jth_hess_residual`](@ref), [`hprod_residual`](@ref), [`hprod_residual!`](@ref), [`hess_op_residual`](@ref), [`hess_op_residual!`](@ref) |


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
jac_coord!
jac_structure
jac_structure!
jac
jac_op
jac_op!
jprod
jprod!
jtprod
jtprod!
hess_coord
hess_coord!
hess_structure
hess_structure!
hess
hess_op
hess_op!
hprod
hprod!
reset!
print
```

## AbstractNLSModel

```@docs
residual
residual!
jac_residual
jac_coord_residual
jac_coord_residual!
jac_structure_residual
jac_structure_residual!
jprod_residual
jprod_residual!
jtprod_residual
jtprod_residual!
jac_op_residual
jac_op_residual!
hess_residual
hess_coord_residual
hess_coord_residual!
hess_structure_residual
hess_structure_residual!
jth_hess_residual
hprod_residual
hprod_residual!
hess_op_residual
hess_op_residual!
```

## Derivative Checker

```@docs
gradient_check
jacobian_check
hessian_check
hessian_check_from_grad
```

## Internal

```@docs
NLPModels.increment!
```
