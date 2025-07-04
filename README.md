# NLPModels

| **Documentation** | **CI** | **Coverage** | **Release** | **DOI** |
|:-----------------:|:------:|:------------:|:-----------:|:-------:|
| [![docs-stable][docs-stable-img]][docs-stable-url] [![docs-dev][docs-dev-img]][docs-dev-url] | [![build-gh][build-gh-img]][build-gh-url] [![build-cirrus][build-cirrus-img]][build-cirrus-url] | [![codecov][codecov-img]][codecov-url] | [![release][release-img]][release-url] | [![doi][doi-img]][doi-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://JuliaSmoothOptimizers.github.io/NLPModels.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://JuliaSmoothOptimizers.github.io/NLPModels.jl/dev
[build-gh-img]: https://github.com/JuliaSmoothOptimizers/NLPModels.jl/workflows/CI/badge.svg?branch=main
[build-gh-url]: https://github.com/JuliaSmoothOptimizers/NLPModels.jl/actions
[build-cirrus-img]: https://img.shields.io/cirrus/github/JuliaSmoothOptimizers/NLPModels.jl?logo=Cirrus%20CI
[build-cirrus-url]: https://cirrus-ci.com/github/JuliaSmoothOptimizers/NLPModels.jl
[codecov-img]: https://codecov.io/gh/JuliaSmoothOptimizers/NLPModels.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaSmoothOptimizers/NLPModels.jl
[release-img]: https://img.shields.io/github/v/release/JuliaSmoothOptimizers/NLPModels.jl.svg?style=flat-square
[release-url]: https://github.com/JuliaSmoothOptimizers/NLPModels.jl/releases
[doi-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.2558627.svg
[doi-url]: https://doi.org/10.5281/zenodo.2558627

This package provides general guidelines to represent non-linear programming (NLP)
problems in Julia and a standardized API to evaluate the functions and
their derivatives. The main objective is to be able to rely on that API when
designing optimization solvers in Julia.

## How to Cite

If you use NLPModels.jl in your work, please cite using the format given in [CITATION.cff](https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/main/CITATION.cff).

## Optimization Problems

Optimization problems are represented by an instance of (a subtype of) `AbstractNLPModel`.
Such instances are composed of
* an instance of `NLPModelMeta`, which provides information about the problem, including the number of variables, constraints, bounds on the variables, etc.
* other data specific to the provenance of the problem.

See the
[documentation](https://JuliaSmoothOptimizers.github.io/NLPModels.jl/dev) for
details on the models and the API.

## Installation

```julia
pkg> add NLPModels
```

## Models

This package provides no models, although it allows the definition of manually written models.

Check the list of packages that define models in [this page of the docs](https://JuliaSmoothOptimizers.github.io/NLPModels.jl/dev/models)

## Main Methods

If `model` is an instance of an appropriate subtype of `AbstractNLPModel`, the following methods are normally defined:

* `obj(model, x)`: evaluate *f(x)*, the objective at `x`
* `cons(model x)`: evaluate *c(x)*, the vector of general constraints at `x`

The following methods are defined if first-order derivatives are available:

* `grad(model, x)`: evaluate *∇f(x)*, the objective gradient at `x`
* `jac(model, x)`: evaluate *J(x)*, the Jacobian of *c* at `x` as a sparse matrix

If Jacobian-vector products can be computed more efficiently than by evaluating the Jacobian explicitly, the following methods may be implemented:

* `jprod(model, x, v)`: evaluate the result of the matrix-vector product *J(x)⋅v*
* `jtprod(model, x, u)`: evaluate the result of the matrix-vector product *J(x)ᵀ⋅u*

The following method is defined if second-order derivatives are available:

* `hess(model, x, y)`: evaluate *∇²L(x,y)*, the Hessian of the Lagrangian at `x` and `y`

If Hessian-vector products can be computed more efficiently than by evaluating the Hessian explicitly, the following method may be implemented:

* `hprod(model, x, v, y)`: evaluate the result of the matrix-vector product *∇²L(x,y)⋅v*

Several in-place variants of the methods above may also be implemented.

The complete list of methods that an interface may implement can be found in the documentation.

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
`nnzh`      | `Int             ` | number of nonzeros in the sparse Hessian
`nnzj`      | `Int             ` | number of nonzeros in the sparse Jacobian
`lin_nnzj`	| `Int             ` | number of nonzeros in the linear part of sparse Jacobian
`nln_nnzj`	| `Int             ` | number of nonzeros in the nonlinear part of sparse Jacobian
`minimize`  | `Bool            ` | true if `optimize == minimize`
`islp`      | `Bool            ` | true if the problem is a linear program
`name`      | `String`           | problem name

# Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/NLPModels.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers), so questions about any of our packages are welcome.
