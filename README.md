# NLPModels

**OSX/Linux:**
master
[![Master Build Status](https://travis-ci.org/JuliaSmoothOptimizers/NLPModels.jl.svg?branch=master)](https://travis-ci.org/JuliaSmoothOptimizers/NLPModels.jl)
develop
[![Develop Build Status](https://travis-ci.org/JuliaSmoothOptimizers/NLPModels.jl.svg?branch=develop)](https://travis-ci.org/JuliaSmoothOptimizers/NLPModels.jl)

**Windows:**
master
[![Master Build status](https://ci.appveyor.com/api/projects/status/l1rs9ajxkyc0cer9/branch/master?svg=true)](https://ci.appveyor.com/project/dpo/nlpmodels-jl/branch/master)
develop
[![Develop Build status](https://ci.appveyor.com/api/projects/status/l1rs9ajxkyc0cer9/branch/develop?svg=true)](https://ci.appveyor.com/project/dpo/nlpmodels-jl/branch/develop)

**Coverage:**
master
[![Master Coverage Status](https://coveralls.io/repos/JuliaSmoothOptimizers/NLPModels.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaSmoothOptimizers/NLPModels.jl?branch=master)
develop
[![Develop Coverage Status](https://coveralls.io/repos/JuliaSmoothOptimizers/NLPModels.jl/badge.svg?branch=develop&service=github)](https://coveralls.io/github/JuliaSmoothOptimizers/NLPModels.jl?branch=develop)

## Objective

This package provides general guidelines to represent optimization problems in Julia and a standardized API to evaluate the functions and their derivatives.
The main objective is to be able to rely on that API when designing optimization solvers in Julia.

## Optimization Problems

Optimization problems are represented by an instance of (a subtype of) `AbstractNLPModel`.
Such instances are composed of
* an instance of `NLPModelMeta`, which provides information about the problem, including the number of variables, constraints, bounds on the variables, etc.
* other data specific to the provenance of the problem.

If the [`JuMP`](https://github.com/JuliaOpt/JuMP.jl) modeling language is available, this package provides the `JuMPNLPModel` type.

Corresponding interfaces are defined in [`AmplNLReader.jl`](https://github.com/JuliaSmoothOptimizers/AmplNLReader.jl) for problems modeled using [AMPL](http://www.ampl.com), and in  [`CUTEst.jl`](https://github.com/JuliaSmoothOptimizers/CUTEst.jl) for problems from [CUTEst](https://ccpforge.cse.rl.ac.uk/gf/project/cutest/wiki).

The `NLPModelMeta` data structure focuses on continuous problems written in the form

    optimize f(x)  subject to l ≤ x ≤ u,  L ≤ c(x) ≤ U,

where `f` is the objective function, `c` is the (vector-valued) constraint function, `l` and `u` are vectors of lower and upper bounds on the variables, and `L` and `U` are vectors of lower and upper bounds on the general constraints.

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

The complete list of methods that an interface may implement is as follows:

* `reset!()`,
* `write_sol()`,
* `varscale()`,
* `lagscale()`,
* `conscale()`,
* `obj()`,
* `grad()`,
* `grad!()`,
* `cons()`,
* `cons!()`,
* `jth_con()`,
* `jth_congrad()`,
* `jth_congrad!()`,
* `jth_sparse_congrad()`,
* `jac_coord()`,
* `jac()`,
* `jth_hprod()`,
* `jth_hprod!()`,
* `ghjvprod()`,
* `ghjvprod!()`,
* `hess_coord()`,
* `hess()`,
* `hprod()`,
* `hprod!`

## Attributes

`NLPModelMeta` objects have the following attributes:

Attribute   | Type               | Notes
------------|--------------------|------------------------------------
`nvar`      | `Int             ` | number of variables
`x0  `      | `Array{Float64,1}` | initial guess
`lvar`      | `Array{Float64,1}` | vector of lower bounds
`uvar`      | `Array{Float64,1}` | vector of upper bounds
`ifix`      | `Array{Int64,1}`   | indices of fixed variables
`ilow`      | `Array{Int64,1}`   | indices of variables with lower bound only
`iupp`      | `Array{Int64,1}`   | indices of variables with upper bound only
`irng`      | `Array{Int64,1}`   | indices of variables with lower and upper bound (range)
`ifree`     | `Array{Int64,1}`   | indices of free variables
`iinf`      | `Array{Int64,1}`   | indices of visibly infeasible bounds
`ncon`      | `Int             ` | total number of general constraints
`nlin `     | `Int             ` | number of linear constraints
`nnln`      | `Int             ` | number of nonlinear general constraints
`nnet`      | `Int             ` | number of nonlinear network constraints
`y0  `      | `Array{Float64,1}` | initial Lagrange multipliers
`lcon`      | `Array{Float64,1}` | vector of constraint lower bounds
`ucon`      | `Array{Float64,1}` | vector of constraint upper bounds
`lin `      | `Range1{Int64}   ` | indices of linear constraints
`nln`       | `Range1{Int64}   ` | indices of nonlinear constraints (not network)
`nnet`      | `Range1{Int64}   ` | indices of nonlinear network constraints
`jfix`      | `Array{Int64,1}`   | indices of equality constraints
`jlow`      | `Array{Int64,1}`   | indices of constraints of the form c(x) ≥ cl
`jupp`      | `Array{Int64,1}`   | indices of constraints of the form c(x) ≤ cu
`jrng`      | `Array{Int64,1}`   | indices of constraints of the form cl ≤ c(x) ≤ cu
`jfree`     | `Array{Int64,1}`   | indices of "free" constraints (there shouldn't be any)
`jinf`      | `Array{Int64,1}`   | indices of the visibly infeasible constraints
`nnzj`      | `Int             ` | number of nonzeros in the sparse Jacobian
`nnzh`      | `Int             ` | number of nonzeros in the sparse Hessian
`minimize`  | `Bool            ` | true if `optimize == minimize`
`islp`      | `Bool            ` | true if the problem is a linear program
`name`      | `ASCIIString     ` | problem name


This content is released under the [MIT](http://opensource.org/licenses/MIT) License.
<a rel="license" href="http://opensource.org/licenses/MIT">
<img alt="MIT license" height="40" src="http://upload.wikimedia.org/wikipedia/commons/c/c3/License_icon-mit.svg" /></a>
