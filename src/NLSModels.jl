export AbstractNLSModel, nls_meta, NLSCounters, nls_counters, reset!,
       residual, residual!, jac_residual, jprod_residual, jprod_residual!,
       jtprod_residual, jtprod_residual!, jac_op_residual, jac_op_residual!,
       hess_residual, hprod_residual, hprod_residual!, hess_op_residual,
       hess_op_residual!, NotImplementedError

abstract AbstractNLSModel <: AbstractNLPModel

type NLSCounters
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

nls_counters(nls :: AbstractNLSModel) = nls.counters

function sum_counters(c :: NLSCounters)
  s = sum_counters(c.counters)
  for field in fieldnames(NLSCounters)
    if field == :counters; continue; end
    s += getfield(c, field)
  end
  return s
end
sum_counters(nls :: AbstractNLSModel) = sum_counters(nls_counters(nls))

for counter in fieldnames(NLSCounters)
  if counter == :counters; continue; end
  @eval begin
    $counter(nls :: AbstractNLSModel) = nls_counters(nls).$counter
    export $counter
  end
end

for counter in fieldnames(Counters)
  @eval begin
    $counter(nls :: AbstractNLSModel) = nls_counters(nls).counters.$counter
    export $counter
  end
end

import NLPModels: reset!

function reset!(nls :: AbstractNLSModel)
  reset!(nls_counters(nls))
  return nls
end

function reset!(nls_counters :: NLSCounters)
  for f in fieldnames(NLSCounters)
    if f == :counters; continue; end
    setfield!(nls_counters, f, 0)
  end
  reset!(nls_counters.counters)
  return counters
end

include("nls_meta.jl")

nls_meta(nls :: AbstractNLSModel) = nls.nls_meta

if Pkg.installed("ForwardDiff") != nothing
  include("autodiff_nlsmodel.jl")
end
include("lls_model.jl")

# min ½‖F(x)‖²

# TODO: Add a regularization ½λ²‖V(x)‖²

"""
    Fx = residual(nls, x)

Computes F(x), the residual at x.
"""
function residual(nls :: AbstractNLSModel, x :: AbstractVector)
  throw(NotImplementedError("residual"))
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
  throw(NotImplementedError("jprod_residual"))
end

"""
    Jv = jprod_residual!(nls, x, v, Jv)

Computes the product of the Jacobian of the residual at x and a vector, i.e.,  J(x)*v, storing it on `Jv`.
"""
function jprod_residual!(nls :: AbstractNLSModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  throw(NotImplementedError("jprod_residual!"))
end

"""
    Jtv = jtprod_residual(nls, x, v)

Computes the product of the transpose of the Jacobian of the residual at x and a vector, i.e.,  J(x)'*v.
"""
function jtprod_residual(nls :: AbstractNLSModel, x :: AbstractVector, v :: AbstractVector)
  throw(NotImplementedError("jtprod_residual"))
end

"""
    Jtv = jtprod_residual!(nls, x, v, Jtv)

Computes the product of the transpose of the Jacobian of the residual at x and a vector, i.e.,  J(x)'*v, storing it on `Jtv`.
"""
function jtprod_residual!(nls :: AbstractNLSModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  throw(NotImplementedError("jtprod_residual!"))
end

"""
    Jx = jac_op_residual(nls, x)

Computes J(x), the Jacobian of the residual at x, on Linear Operator form.
"""
function jac_op_residual(nls :: AbstractNLSModel, x :: AbstractVector)
  return LinearOperator{Float64}(nls_meta(nls).nequ, nls_meta(nls).nvar,
                                 false, false,
                                 v -> jprod_residual(nls, x, v),
                                 Nullable{Function}(),
                                 v -> jtprod_residual(nls, x, v))
end

"""
    Jx = jac_op_residual!(nls, x, Jv, Jtv)

Computes J(x), the Jacobian of the residual at x, on Linear Operator form. The
vectors `Jv` and `Jtv` are used as preallocated storage for the operations.
"""
function jac_op_residual!(nls :: AbstractNLSModel, x :: AbstractVector,
                          Jv :: AbstractVector, Jtv :: AbstractVector)
  return LinearOperator{Float64}(nls_meta(nls).nequ, nls_meta(nls).nvar,
                                 false, false,
                                 v -> jprod_residual!(nls, x, v, Jv),
                                 Nullable{Function}(),
                                 v -> jtprod_residual!(nls, x, v, Jtv))
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
  throw(NotImplementedError("hprod_residual"))
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

Computes the Hessian of the i-th residual at x, in LinearOperator form.
"""
function hess_op_residual(nls :: AbstractNLSModel, x :: AbstractVector, i :: Int)
  return LinearOperator(nls_meta(nls).nvar, nls_meta(nls).nvar,
                        true, true,
                        v -> hprod_residual(nls, x, i, v))
end

"""
    Hop = hess_op_residual!(nls, x, i, Hiv)

Computes the Hessian of the i-th residual at x, in LinearOperator form. The vector `Hiv` is used as preallocated storage for the operation.
"""
function hess_op_residual!(nls :: AbstractNLSModel, x :: AbstractVector, i :: Int, Hiv :: AbstractVector)
  return LinearOperator(nls_meta(nls).nvar, nls_meta(nls).nvar,
                        true, true,
                        v -> hprod_residual!(nls, x, i, v, Hiv))
end

function obj(nls :: AbstractNLSModel, x :: AbstractVector)
  nls_counters(nls).counters.neval_obj += 1
  Fx = residual(nls, x)
  return 0.5*dot(Fx, Fx)
end

function grad(nls :: AbstractNLSModel, x :: AbstractVector)
  nls_counters(nls).counters.neval_grad += 1
  Fx = residual(nls, x)
  return jtprod_residual(nls, x, Fx)
end

function grad!(nls :: AbstractNLSModel, x :: AbstractVector, g :: AbstractVector)
  nls_counters(nls).counters.neval_grad += 1
  Fx = residual(nls, x)
  return jtprod_residual!(nls, x, Fx, g)
end

function objgrad(nls :: AbstractNLSModel, x :: AbstractVector)
  nls_counters(nls).counters.neval_obj += 1
  nls_counters(nls).counters.neval_grad += 1
  Fx = residual(nls, x)
  g = jtprod_residual(nls, x, Fx)
  return 0.5*dot(Fx, Fx), g
end

function objgrad!(nls :: AbstractNLSModel, x :: AbstractVector, g :: AbstractVector)
  nls_counters(nls).counters.neval_obj += 1
  nls_counters(nls).counters.neval_grad += 1
  Fx = residual(nls, x)
  jtprod_residual!(nls, x, Fx, g)
  return 0.5*dot(Fx, Fx), g
end

function hess(nls :: AbstractNLSModel, x :: AbstractVector; obj_weight ::
              Float64 = 1.0, y :: AbstractVector = Float64[])
  nls_counters(nls).counters.neval_hess += 1
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
  nls_counters(nls).counters.neval_hprod += 1
  Fx = residual(nls, x)
  Hv = jtprod_residual(nls, x, jprod_residual(nls, x, v))
  m = length(Fx)
  for i = 1:m
    Hv += Fx[i] * hprod_residual(nls, x, i, v)
  end
  return obj_weight * Hv
end

function hprod!(nls :: AbstractNLSModel, x :: AbstractVector, v ::
                AbstractVector, Hv :: AbstractVector; obj_weight :: Float64 =
                1.0, y :: AbstractVector = Float64[])
  nls_counters(nls).counters.neval_hprod += 1
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
