module NLPModels

# stdlib
using LinearAlgebra, Printf, SparseArrays
# external
using FastClosures
# JSO
using LinearOperators

export AbstractNLPModel, AbstractNLSModel, AbstractDenseNLPModel

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

"""
    AbstractNLPModel

Base type for an optimization model.
"""
abstract type AbstractNLPModel{T, S} end

"""
    AbstractDenseNLPModel <: AbstractNLPModel

Base type for a dense optimization model (Jacobian/Hessian stored as dense matrices).
"""
abstract type AbstractDenseNLPModel{T, S} <: AbstractNLPModel{T, S} end

"""
    AbstractNLSModel <: AbstractNLPModel

Base type for a nonlinear least-squares model.
"""
abstract type AbstractNLSModel{T, S} <: AbstractNLPModel{T, S} end

for f in ["utils", "api", "counters", "meta", "show", "tools"]
  include("nlp/$f.jl")
  include("nls/$f.jl")
end

end # module
