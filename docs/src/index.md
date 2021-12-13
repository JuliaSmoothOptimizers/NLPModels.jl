# [NLPModels.jl documentation](@id Home)

This package provides general guidelines to represent optimization problems in
Julia and a standardized API to evaluate the functions and their derivatives.
The main objective is to be able to rely on that [API](@ref) when designing
optimization solvers in Julia.

## Introduction

The general form of the optimization problem is
```math
\begin{aligned}
\min \quad & f(x) \\
& c_i(x) = 0, \quad i \in E, \\
& c_{L_i} \leq c_i(x) \leq c_{U_i}, \quad i \in I, \\
& \ell \leq x \leq u,
\end{aligned}
```
where ``f:\mathbb{R}^n\rightarrow\mathbb{R}``,
``c:\mathbb{R}^n\rightarrow\mathbb{R}^m``,
``E\cup I = \{1,2,\dots,m\}``, ``E\cap I = \emptyset``,
and
``c_{L_i}, c_{U_i}, \ell_j, u_j \in \mathbb{R}\cup\{\pm\infty\}``
for ``i = 1,\dots,m`` and ``j = 1,\dots,n``.

For computational reasons, we write
```math
\begin{aligned}
\min \quad & f(x) \\
& c_L \leq c(x) \leq c_U \\
& \ell \leq x \leq u,
\end{aligned}
```
defining ``c_{L_i} = c_{U_i}`` for all ``i \in E``.
The Lagrangian of this problem is defined as
```math
L(x,\lambda,z^L,z^U;\sigma) = \sigma f(x) + c(x)^T\lambda  + \sum_{i=1}^n z_i^L(x_i-l_i) + \sum_{i=1}^nz_i^U(u_i-x_i),
```
where ``\sigma`` is a scaling parameter included for computational reasons.
Notice that, for the Hessian, the variables ``z^L`` and ``z^U`` are not used.

Optimization problems are represented by an instance/subtype of `AbstractNLPModel`.
Such instances are composed of

- an instance of [`NLPModelMeta`](@ref Attributes), which provides information about the problem,
  including the number of variables, constraints, bounds on the variables, etc.
- other data specific to the provenance of the problem.

## Nonlinear Least Squares

A special type of `NLPModels` are the `NLSModels`, i.e., Nonlinear Least
Squares models. In these problems, the function ``f(x)`` is given by
``\tfrac{1}{2}\Vert F(x)\Vert^2``, where ``F`` is referred as the residual function.
The individual value of ``F``, as well as of its derivatives, is also
available.

## Tools

There are a few tools to use on `NLPModels`, for instance to query
whether the problem is constrained or not, and to get the number of
function evaluations. See [Tools](@ref tools-section).

## Install

Install NLPModels.jl with the following command.
```julia
pkg> add NLPModels
```

This will enable the use of the API and the tools described here, and it allows the creation of a manually written model.
Look into [Models](@ref) for more information on that subject, and on a list of packages implementing ready-to-use models.

## Usage

See the [Models](@ref), the [Tools](@ref tools-section), or the [API](@ref).


## Attributes

`NLPModelMeta` objects have the following attributes (with `S <: AbstractVector`):

Attribute   | Type               | Notes
------------|--------------------|------------------------------------
`nvar`      | `Int             ` | number of variables
`x0  `      | `S`                | initial guess
`lvar`      | `S`                | vector of lower bounds
`uvar`      | `S`                | vector of upper bounds
`ifix`      | `Vector{Int}`      | indices of fixed variables
`ilow`      | `Vector{Int}`      | indices of variables with lower bound only
`iupp`      | `Vector{Int}`      | indices of variables with upper bound only
`irng`      | `Vector{Int}`      | indices of variables with lower and upper bound (range)
`ifree`     | `Vector{Int}`      | indices of free variables
`iinf`      | `Vector{Int}`      | indices of visibly infeasible bounds
`ncon`      | `Int             ` | total number of general constraints
`nlin `     | `Int             ` | number of linear constraints
`nnln`      | `Int             ` | number of nonlinear general constraints
`y0  `      | `S`                | initial Lagrange multipliers
`lcon`      | `S`                | vector of constraint lower bounds
`ucon`      | `S`                | vector of constraint upper bounds
`lin `      | `Vector{Int}`      | indices of linear constraints
`nln`       | `Vector{Int}`      | indices of nonlinear constraints
`jfix`      | `Vector{Int}`      | indices of equality constraints
`jlow`      | `Vector{Int}`      | indices of constraints of the form c(x) ≥ cl
`jupp`      | `Vector{Int}`      | indices of constraints of the form c(x) ≤ cu
`jrng`      | `Vector{Int}`      | indices of constraints of the form cl ≤ c(x) ≤ cu
`jfree`     | `Vector{Int}`      | indices of "free" constraints (there shouldn't be any)
`jinf`      | `Vector{Int}`      | indices of the visibly infeasible constraints
`nnzo`      | `Int             ` | number of nonzeros in the gradient
`nnzj`      | `Int             ` | number of nonzeros in the sparse Jacobian
`lin_nnzj`  | `Int             ` | number of nonzeros in the sparse linear constraints Jacobian
`nln_nnzj`  | `Int             ` | number of nonzeros in the sparse nonlinear constraints Jacobian
`nnzh`      | `Int             ` | number of nonzeros in the sparse Hessian
`minimize`  | `Bool            ` | true if `optimize == minimize`
`islp`      | `Bool            ` | true if the problem is a linear program
`name`      | `String`           | problem name

## License

This content is released under the [MPL2.0](https://www.mozilla.org/en-US/MPL/2.0/) License.

## Contents

```@contents
```
