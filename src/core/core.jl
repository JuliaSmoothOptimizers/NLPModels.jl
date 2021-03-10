export AbstractNLPModel, AbstractNLSModel

# For documentation purpose
const OBJECTIVE_HESSIAN = raw"""
```math
σ ∇²f(x),
```
with `σ = obj_weight`
"""
const LAGRANGIAN_HESSIAN = raw"""
```math
∇²L(x,y) = σ ∇²f(x) + \sum_i yᵢ ∇²cᵢ(x),
```
with `σ = obj_weight`
"""

# Base type for an optimization model.
abstract type AbstractNLPModel end

abstract type AbstractNLSModel <: AbstractNLPModel end

for f in ["utils", "api", "counters", "meta", "show", "tools", ]
  include("nlp/$f.jl")
  include("nls/$f.jl")
end