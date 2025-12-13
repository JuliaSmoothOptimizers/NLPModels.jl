# API

As stated in the [Home](@ref) page, we consider the nonlinear optimization
problem in the following format:
```math
\begin{aligned}
\min \quad & f(x) \\
& c_L \leq c(x) \leq c_U \\
& \ell \leq x \leq u.
\end{aligned}
```
To develop an optimization algorithm, we are usually worried not only with
``f(x)`` and ``c(x)``, but also with their derivatives.
Namely,

- ``\nabla f(x)``, the gradient of ``f`` at the point ``x``;
- ``\nabla^2 f(x)``, the Hessian of ``f`` at the point ``x``;
- ``J(x) = \nabla c(x)^T``, the Jacobian of ``c`` at the point ``x``;
- ``\nabla^2 f(x) + \sum_{i=1}^m y_i \nabla^2 c_i(x)``,
  the Hessian of the Lagrangian function at the point ``(x,y)``.

There are many ways to access some of these values, so here is a little
reference guide.

## Reference guide

The following naming should be easy enough to follow.
If not, click on the link and go to the description.

- `!` means inplace;
- `_coord` means coordinate format;
- `prod` means matrix-vector product;
- `_op` means operator (as in [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl));
- `_lin` and `_nln` respectively refer to linear and nonlinear constraints.

Feel free to open an issue to suggest other methods that should apply to all
NLPModels instances.

| Function          | NLPModels function                                                                                                                                                                                   |
|-------------------|-------------------------------------------|
| ``f(x)``            | [`obj`](@ref), [`objgrad`](@ref), [`objgrad!`](@ref), [`objcons`](@ref), [`objcons!`](@ref) |
| ``\nabla f(x)``     | [`grad`](@ref), [`grad!`](@ref), [`objgrad`](@ref), [`objgrad!`](@ref) |
| ``\nabla^2 f(x)``   | [`hess`](@ref), [`hess_op`](@ref), [`hess_op!`](@ref), [`hess_coord`](@ref), [`hess_coord`](@ref), [`hess_structure`](@ref), [`hess_structure!`](@ref), [`hprod`](@ref), [`hprod!`](@ref) |
| ``c(x)``            | [`cons_lin`](@ref), [`cons_lin!`](@ref), [`cons_nln`](@ref), [`cons_nln!`](@ref), [`cons`](@ref), [`cons!`](@ref), [`objcons`](@ref), [`objcons!`](@ref) |
| ``J(x)``            | [`jac_lin`](@ref), [`jac_nln`](@ref), [`jac`](@ref), [`jac_lin_op`](@ref), [`jac_lin_op!`](@ref), [`jac_nln_op`](@ref), [`jac_nln_op!`](@ref),[`jac_op`](@ref), [`jac_op!`](@ref), [`jac_lin_coord`](@ref), [`jac_lin_coord!`](@ref), [`jac_nln_coord`](@ref), [`jac_nln_coord!`](@ref), [`jac_coord`](@ref), [`jac_coord!`](@ref), [`jac_lin_structure`](@ref), [`jac_lin_structure!`](@ref), [`jac_nln_structure`](@ref), [`jac_nln_structure!`](@ref), [`jac_structure`](@ref), [`jprod_lin`](@ref), [`jprod_lin!`](@ref), [`jprod_nln`](@ref), [`jprod_nln!`](@ref), [`jprod`](@ref), [`jprod!`](@ref), [`jtprod_lin`](@ref), [`jtprod_lin!`](@ref), [`jtprod_nln`](@ref), [`jtprod_nln!`](@ref), [`jtprod`](@ref), [`jtprod!`](@ref) |
| ``\nabla^2 L(x,y)`` | [`hess`](@ref), [`hess_op`](@ref), [`hess_coord`](@ref), [`hess_coord!`](@ref), [`hess_structure`](@ref), [`hess_structure!`](@ref), [`hprod`](@ref), [`hprod!`](@ref), [`jth_hprod`](@ref), [`jth_hprod!`](@ref), [`jth_hess`](@ref), [`jth_hess_coord`](@ref), [`jth_hess_coord!`](@ref), [`ghjvprod`](@ref), [`ghjvprod!`](@ref)  |

If only a subset of the functions listed above is implemented, you can indicate which ones are not available when creating the [`NLPModelMeta`](@ref), using the keyword arguments
`grad_available`, `jac_available`, `hess_available`, `jprod_available`, `jtprod_available`, and `hprod_available`.

## [API for NLSModels](@id nls-api)

For the Nonlinear Least Squares models, ``f(x) = \tfrac{1}{2} \Vert F(x)\Vert^2``,
and these models have additional function to access the residual value
and its derivatives. Namely,

- ``J_F(x) = \nabla F(x)^T``
- ``\nabla^2 F_i(x)``

| Function            | function |
|---------------------|---|
| ``F(x)``            | [`residual`](@ref), [`residual!`](@ref) |
| ``J_F(x)``          | [`jac_residual`](@ref), [`jac_coord_residual`](@ref), [`jac_coord_residual!`](@ref), [`jac_structure_residual`](@ref), [`jac_structure_residual!`](@ref), [`jprod_residual`](@ref), [`jprod_residual!`](@ref), [`jtprod_residual`](@ref), [`jtprod_residual!`](@ref), [`jac_op_residual`](@ref), [`jac_op_residual!`](@ref) |
| ``\nabla^2 F_i(x)`` | [`hess_residual`](@ref), [`hess_coord_residual`](@ref), [`hess_coord_residual!`](@ref), [`hess_structure_residual`](@ref), [`hess_structure_residual!`](@ref), [`jth_hess_residual`](@ref), [`jth_hess_residual_coord`](@ref), [`jth_hess_residual_coord!`](@ref), [`hprod_residual`](@ref), [`hprod_residual!`](@ref), [`hess_op_residual`](@ref), [`hess_op_residual!`](@ref) |

If only a subset of the functions listed above is implemented, you can indicate which ones are not available when creating the [`NLSMeta`](@ref), using the keyword arguments `jac_residual_available`, `hess_residual_available`, `jprod_residual_available`, `jtprod_residual_available`, and `hprod_residual_available`.
