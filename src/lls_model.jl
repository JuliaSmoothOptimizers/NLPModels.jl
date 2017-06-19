export LLSModel,
       residual, residual!, jac_residual, jprod_residual, jprod_residual!,
       jtprod_residual, jtprod_residual!, jac_op_residual, hess_residual,
       hprod_residual, hprod_residual!

"""
    nls = LLSModel(A, b)

Creates a Linear Least Squares model ½‖Ax - b‖²
"""
type LLSModel <: AbstractNLSModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  counters :: NLSCounters

  A :: Union{AbstractMatrix, LinearOperator}
  b :: AbstractVector
end

function LLSModel(A :: Union{AbstractMatrix, LinearOperator}, b :: AbstractVector, λ :: Real = 0.0)
  m, n = size(A)
  if length(b) != m
    error("Incompatibility detected: A is $m×$n and b has lenght $(length(b))")
  end

  meta = NLPModelMeta(n, x0=zeros(n))
  nls_meta = NLSMeta(m, n)

  return LLSModel(meta, nls_meta, NLSCounters(), A, b)
end

function residual(nls :: LLSModel, x :: AbstractVector)
  nls.counters.neval_residual += 1
  return nls.A * x - nls.b
end

function residual!(nls :: LLSModel, x :: AbstractVector, Fx :: AbstractVector)
  nls.counters.neval_residual += 1
  Fx[:] = nls.A * x - nls.b
  return Fx
end

function jac_residual(nls :: LLSModel, x :: AbstractVector)
  nls.counters.neval_jac_residual += 1
  if isa(nls.A, LinearOperator)
    error("Jacobian is a LinearOperator. Use `jac_op_residual` instead.")
  else
    return nls.A
  end
end

function jprod_residual(nls :: LLSModel, x :: AbstractVector, v :: AbstractVector)
  nls.counters.neval_jprod_residual += 1
  return nls.A * v
end

function jprod_residual!(nls :: LLSModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  nls.counters.neval_jprod_residual += 1
  Jv[:] = nls.A * v
  return Jv
end

function jtprod_residual(nls :: LLSModel, x :: AbstractVector, v :: AbstractVector)
  nls.counters.neval_jtprod_residual += 1
  return nls.A' * v
end

function jtprod_residual!(nls :: LLSModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  nls.counters.neval_jtprod_residual += 1
  Jtv[:] = nls.A' * v
  return Jtv
end

function hess_residual(nls :: LLSModel, x :: AbstractVector, i :: Int)
  nls.counters.neval_hess_residual += 1
  n = size(nls.A, 2)
  return zeros(n, n)
end

function hprod_residual(nls :: LLSModel, x :: AbstractVector, i :: Int, v :: AbstractVector)
  nls.counters.neval_hprod_residual += 1
  n = size(nls.A, 2)
  return zeros(n)
end

function hprod_residual!(nls :: LLSModel, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  nls.counters.neval_hprod_residual += 1
  fill!(Hiv, 0.0)
  return Hiv
end
