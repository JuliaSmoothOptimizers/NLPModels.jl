export LLSModel,
       residual!, jac_residual, jprod_residual!, jtprod_residual!,
       jac_op_residual, hess_residual, hprod_residual!

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

function LLSModel(A :: Union{AbstractMatrix, LinearOperator}, b :: AbstractVector;
                  x0 :: Vector = zeros(size(A,2)),
                  lvar :: Vector = fill(-Inf, size(A, 2)),
                  uvar :: Vector = fill(Inf, size(A, 2)))
  m, n = size(A)
  if length(b) != m
    error("Incompatibility detected: A is $m×$n and b has lenght $(length(b))")
  end

  meta = NLPModelMeta(n, x0=x0, lvar=lvar, uvar=uvar)
  nls_meta = NLSMeta(m, n)

  return LLSModel(meta, nls_meta, NLSCounters(), A, b)
end

function residual!(nls :: LLSModel, x :: AbstractVector, Fx :: AbstractVector)
  increment!(nls, :neval_residual)
  Fx[:] = nls.A * x - nls.b
  return Fx
end

function jac_residual(nls :: LLSModel, x :: AbstractVector)
  increment!(nls, :neval_jac_residual)
  if isa(nls.A, LinearOperator)
    error("Jacobian is a LinearOperator. Use `jac_op_residual` instead.")
  else
    return nls.A
  end
end

function jprod_residual!(nls :: LLSModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nls, :neval_jprod_residual)
  Jv[:] = nls.A * v
  return Jv
end

function jtprod_residual!(nls :: LLSModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nls, :neval_jtprod_residual)
  Jtv[:] = nls.A' * v
  return Jtv
end

function hess_residual(nls :: LLSModel, x :: AbstractVector, i :: Int)
  increment!(nls, :neval_hess_residual)
  n = size(nls.A, 2)
  return zeros(n, n)
end

function hprod_residual!(nls :: LLSModel, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  increment!(nls, :neval_hprod_residual)
  fill!(Hiv, 0.0)
  return Hiv
end
