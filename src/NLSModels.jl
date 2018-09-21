export AbstractNLSModel, nls_meta, NLSCounters, reset!,
       residual, residual!, jac_residual, jprod_residual, jprod_residual!,
       jtprod_residual, jtprod_residual!, jac_op_residual, jac_op_residual!,
       hess_residual, hprod_residual, hprod_residual!, hess_op_residual,
       hess_op_residual!, NotImplementedError

abstract type AbstractNLSModel <: AbstractNLPModel end

mutable struct NLSCounters
  counters :: Counters
  neval_residual        :: Int
  neval_jac_residual    :: Int
  neval_jprod_residual  :: Int
  neval_jtprod_residual :: Int
  neval_hess_residual   :: Int
  neval_hprod_residual  :: Int

  function NLSCounters()
    return new(Counters(), 0, 0, 0, 0, 0, 0)
  end
end

function increment!(nls :: AbstractNLSModel, s :: Symbol)
  if s in fieldnames(NLSCounters)
    setfield!(nls.counters, s, getfield(nls.counters, s) + 1)
  else
    setfield!(nls.counters.counters, s, getfield(nls.counters.counters, s) + 1)
  end
end

function sum_counters(c :: NLSCounters)
  s = sum_counters(c.counters)
  for field in fieldnames(NLSCounters)
    field == :counters && continue
    s += getfield(c, field)
  end
  return s
end
sum_counters(nls :: AbstractNLSModel) = sum_counters(nls.counters)

for counter in fieldnames(NLSCounters)
  counter == :counters && continue
  @eval begin
    """`$($counter)(nlp)`

    Get the number of `$(split("$($counter)", "_")[2])` evaluations.
    """
    $counter(nls :: AbstractNLSModel) = nls.counters.$counter
    export $counter
  end
end

for counter in fieldnames(Counters)
  @eval begin
    $counter(nls :: AbstractNLSModel) = nls.counters.counters.$counter
    export $counter
  end
end

import NLPModels: reset!

function reset!(nls :: AbstractNLSModel)
  reset!(nls.counters)
  return nls
end

function reset!(nls_counters :: NLSCounters)
  for f in fieldnames(NLSCounters)
    f == :counters && continue
    setfield!(nls_counters, f, 0)
  end
  reset!(nls_counters.counters)
  return nls_counters
end

include("nls_meta.jl")

nls_meta(nls :: AbstractNLSModel) = nls.nls_meta

include("autodiff_nlsmodel.jl")
include("lls_model.jl")
include("simple_nls_model.jl")
include("feasibility_residual.jl")

# min ½‖F(x)‖²

# TODO: Add a regularization ½λ²‖V(x)‖²

"""
    Fx = residual(nls, x)

Computes F(x), the residual at x.
"""
function residual(nls :: AbstractNLSModel, x :: AbstractVector)
  Fx = zeros(nls_meta(nls).nequ)
  residual!(nls, x, Fx)
end

"""
    Fx = residual!(nls, x, Fx)

Computes F(x), the residual at x.
"""
function residual!(nls :: AbstractNLSModel, x :: AbstractVector, Fx :: AbstractVector)
  throw(NotImplementedError("residual!"))
end

"""
    Jx = jac_residual(nls, x)

Computes J(x), the Jacobian of the residual at x.
"""
function jac_residual(nls :: AbstractNLSModel, x :: AbstractVector)
  throw(NotImplementedError("jac_residual"))
end

"""
    Jv = jprod_residual(nls, x, v)

Computes the product of the Jacobian of the residual at x and a vector, i.e.,  J(x)*v.
"""
function jprod_residual(nls :: AbstractNLSModel, x :: AbstractVector, v :: AbstractVector)
  Jv = zeros(nls_meta(nls).nequ)
  jprod_residual!(nls, x, v, Jv)
end

"""
    Jv = jprod_residual!(nls, x, v, Jv)

Computes the product of the Jacobian of the residual at x and a vector, i.e.,  J(x)*v, storing it in `Jv`.
"""
function jprod_residual!(nls :: AbstractNLSModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  throw(NotImplementedError("jprod_residual!"))
end

"""
    Jtv = jtprod_residual(nls, x, v)

Computes the product of the transpose of the Jacobian of the residual at x and a vector, i.e.,  J(x)'*v.
"""
function jtprod_residual(nls :: AbstractNLSModel, x :: AbstractVector, v :: AbstractVector)
  Jtv = zeros(nls_meta(nls).nvar)
  jtprod_residual!(nls, x, v, Jtv)
end

"""
    Jtv = jtprod_residual!(nls, x, v, Jtv)

Computes the product of the transpose of the Jacobian of the residual at x and a vector, i.e.,  J(x)'*v, storing it in `Jtv`.
"""
function jtprod_residual!(nls :: AbstractNLSModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  throw(NotImplementedError("jtprod_residual!"))
end

"""
    Jx = jac_op_residual(nls, x)

Computes J(x), the Jacobian of the residual at x, in linear operator form.
"""
function jac_op_residual(nls :: AbstractNLSModel, x :: AbstractVector)
  prod = v -> jprod_residual(nls, x, v)
  ctprod = v -> jtprod_residual(nls, x, v)
  F1 = typeof(prod)
  F3 = typeof(ctprod)
  return LinearOperator{Float64,F1,Nothing,F3}(nls_meta(nls).nequ, nls_meta(nls).nvar,
                                               false, false, prod, nothing, ctprod)
end

"""
    Jx = jac_op_residual!(nls, x, Jv, Jtv)

Computes J(x), the Jacobian of the residual at x, in linear operator form. The
vectors `Jv` and `Jtv` are used as preallocated storage for the operations.
"""
function jac_op_residual!(nls :: AbstractNLSModel, x :: AbstractVector,
                          Jv :: AbstractVector, Jtv :: AbstractVector)
  prod = v -> jprod_residual!(nls, x, v, Jv)
  ctprod = v -> jtprod_residual!(nls, x, v, Jtv)
  F1 = typeof(prod)
  F3 = typeof(ctprod)
  return LinearOperator{Float64,F1,Nothing,F3}(nls_meta(nls).nequ, nls_meta(nls).nvar,
                                               false, false, prod, nothing, ctprod)
end

"""
    Hi = hess_residual(nls, x, i)

Computes the Hessian of the i-th residual at x.
"""
function hess_residual(nls :: AbstractNLSModel, x :: AbstractVector, i :: Int)
  throw(NotImplementedError("hess_residual"))
end

"""
    Hiv = hprod_residual(nls, x, i, v)

Computes the product of the Hessian of the i-th residual at x, times the vector v.
"""
function hprod_residual(nls :: AbstractNLSModel, x :: AbstractVector, i :: Int, v :: AbstractVector)
  Hv = zeros(nls_meta(nls).nvar)
  hprod_residual!(nls, x, i, v, Hv)
end

"""
    Hiv = hprod_residual!(nls, x, i, v, Hiv)

Computes the product of the Hessian of the i-th residual at x, times the vector v, and stores it in vector Hiv.
"""
function hprod_residual!(nls :: AbstractNLSModel, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  throw(NotImplementedError("hprod_residual!"))
end

"""
    Hop = hess_op_residual(nls, x, i)

Computes the Hessian of the i-th residual at x, in linear operator form.
"""
function hess_op_residual(nls :: AbstractNLSModel, x :: AbstractVector, i :: Int)
  prod = v -> hprod_residual(nls, x, i, v)
  F = typeof(prod)
  return LinearOperator{Float64,F,Nothing,Nothing}(nls_meta(nls).nvar, nls_meta(nls).nvar,
                                                   true, true, prod, nothing, nothing)
end

"""
    Hop = hess_op_residual!(nls, x, i, Hiv)

Computes the Hessian of the i-th residual at x, in linear operator form. The vector `Hiv` is used as preallocated storage for the operation.
"""
function hess_op_residual!(nls :: AbstractNLSModel, x :: AbstractVector, i :: Int, Hiv :: AbstractVector)
  prod = v -> hprod_residual!(nls, x, i, v, Hiv)
  F = typeof(prod)
  return LinearOperator{Float64,F,Nothing,Nothing}(nls_meta(nls).nvar, nls_meta(nls).nvar,
                                                   true, true, prod, nothing, nothing)
end

function obj(nls :: AbstractNLSModel, x :: AbstractVector)
  increment!(nls, :neval_obj)
  Fx = residual(nls, x)
  return 0.5*dot(Fx, Fx)
end

function grad(nls :: AbstractNLSModel, x :: AbstractVector)
  g = zeros(nls_meta(nls).nvar)
  return grad!(nls, x, g)
end

function grad!(nls :: AbstractNLSModel, x :: AbstractVector, g :: AbstractVector)
  increment!(nls, :neval_grad)
  Fx = residual(nls, x)
  return jtprod_residual!(nls, x, Fx, g)
end

function objgrad(nls :: AbstractNLSModel, x :: AbstractVector)
  g = zeros(nls_meta(nls).nvar)
  return objgrad!(nls, x, g)
end

function objgrad!(nls :: AbstractNLSModel, x :: AbstractVector, g :: AbstractVector)
  increment!(nls, :neval_obj)
  increment!(nls, :neval_grad)
  Fx = residual(nls, x)
  jtprod_residual!(nls, x, Fx, g)
  return 0.5*dot(Fx, Fx), g
end

function hess(nls :: AbstractNLSModel, x :: AbstractVector; obj_weight ::
              Float64 = 1.0, y :: AbstractVector = Float64[])
  increment!(nls, :neval_hess)
  Fx = residual(nls, x)
  Jx = jac_residual(nls, x)
  Hx = tril(Jx'*Jx)
  m = length(Fx)
  for i = 1:m
    Hx += Fx[i] * hess_residual(nls, x, i)
  end
  return obj_weight * Hx
end

function hess_coord(nls :: AbstractNLSModel, x :: AbstractVector; obj_weight ::
                    Float64 = 1.0, y :: AbstractVector = Float64[])
  Hx = hess(nls, x, obj_weight=obj_weight)
  return findnz(Hx)
end

function hprod(nls :: AbstractNLSModel, x :: AbstractVector, v ::
               AbstractVector; obj_weight :: Float64 = 1.0, y :: AbstractVector
               = Float64[])
  Hv = zeros(nls_meta(nls).nvar)
  return hprod!(nls, x, v, Hv, obj_weight=obj_weight, y=y)
end

function hprod!(nls :: AbstractNLSModel, x :: AbstractVector, v ::
                AbstractVector, Hv :: AbstractVector; obj_weight :: Float64 =
                1.0, y :: AbstractVector = Float64[])
  increment!(nls, :neval_hprod)
  Fx = residual(nls, x)
  Jv = jprod_residual(nls, x, v)
  jtprod_residual!(nls, x, Jv, Hv)
  m = length(Fx)
  Hiv = zeros(length(x))
  for i = 1:m
    hprod_residual!(nls, x, i, v, Hiv)
    Hv .= Hv .+ Fx[i] * Hiv
  end
  return obj_weight * Hv
end
