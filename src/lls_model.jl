export LLSModel,
       residual!, jac_residual, jprod_residual!, jtprod_residual!,
       jac_op_residual, hess_residual, hprod_residual!

"""
    nls = LLSModel(A, b; lvar, uvar, C, lcon, ucon)

Creates a Linear Least Squares model ½‖Ax - b‖² with optional bounds
`lvar ≦ x ≦ y` and optional linear constraints `lcon ≦ Cx ≦ ucon`.
"""
mutable struct LLSModel <: AbstractNLSModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  counters :: NLSCounters

  A :: Union{AbstractMatrix, LinearOperator}
  b :: AbstractVector
  C :: Union{AbstractMatrix, LinearOperator}
end

function LLSModel(A :: Union{AbstractMatrix, LinearOperator}, b :: AbstractVector;
                  x0 :: AbstractVector = zeros(size(A,2)),
                  lvar :: AbstractVector = fill(-Inf, size(A, 2)),
                  uvar :: AbstractVector = fill(Inf, size(A, 2)),
                  C :: Union{AbstractMatrix, LinearOperator} = Matrix{Float64}(undef, 0, 0),
                  lcon :: AbstractVector = Float64[],
                  ucon :: AbstractVector = Float64[],
                  y0 :: AbstractVector = zeros(size(C,1)))
  m, n = size(A)
  if length(b) != m
    error("Incompatibility detected: A is $m×$n and b has length $(length(b))")
  end
  ncon = size(C, 1)
  if !(ncon == length(lcon) == length(ucon) == length(y0))
    error("The number of rows in C must be the same length as lcon, ucon and y0")
  end
  nnzj = n * ncon

  meta = NLPModelMeta(n, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, y0=y0, lin=1:ncon,
                      nln=Int[], lcon=lcon, ucon=ucon, nnzj=nnzj)
  nls_meta = NLSMeta(m, n)

  return LLSModel(meta, nls_meta, NLSCounters(), A, b, C)
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

function cons(nls :: LLSModel, x :: AbstractVector)
  increment!(nls, :neval_cons)
  return nls.C * x
end

function cons!(nls :: LLSModel, x :: AbstractVector, c :: AbstractVector)
  increment!(nls, :neval_cons)
  c[1:nls.meta.ncon] = nls.C * x
  return c
end

function jac_coord(nls :: LLSModel, x :: AbstractVector)
  increment!(nls, :neval_jac)
  if isa(nls.C, LinearOperator)
    error("jac_coord is not defined for LinearOperators")
  end
  if isa(nls.C, SparseMatrixCSC)
    return findnz(nls.C)
  else
    I = findall(!iszero, nls.C)
    return (getindex.(I, 1), getindex.(I, 2), nls.C[I])
  end
end

function jac(nls :: LLSModel, x :: AbstractVector)
  increment!(nls, :neval_jac)
  return nls.C
end

function jprod(nls :: LLSModel, x :: AbstractVector, v :: AbstractVector)
  increment!(nls, :neval_jprod)
  return nls.C * v
end

function jprod!(nls :: LLSModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nls, :neval_jprod)
  Jv[1:nls.meta.ncon] = nls.C * v
  return Jv
end

function jtprod(nls :: LLSModel, x :: AbstractVector, v :: AbstractVector)
  increment!(nls, :neval_jtprod)
  return nls.C' * v
end

function jtprod!(nls :: LLSModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nls, :neval_jtprod)
  Jtv[1:nls.meta.nvar] = nls.C' * v
  return Jtv
end

function hess(nls :: LLSModel, x :: AbstractVector; obj_weight = 1.0, y :: AbstractVector = Float64[])
  increment!(nls, :neval_hess)
  if obj_weight != 0.0
    if isa(nls.A, LinearOperator)
      error("hess is not defined for LinearOperators")
    end
    return tril(obj_weight * (nls.A' * nls.A))
  else
    n = length(x)
    return zeros(n, n)
  end
end

function hprod(nls :: LLSModel, x :: AbstractVector, v :: AbstractVector;
    obj_weight = 1.0, y :: AbstractVector = Float64[])
  Hv = zeros(nls.meta.nvar)
  return hprod!(nls, x, v, Hv, obj_weight=obj_weight, y=y)
end

function hprod!(nls :: LLSModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector;
    obj_weight = 1.0, y :: AbstractVector = Float64[])
  increment!(nls, :neval_hprod)
  n = length(x)
  if obj_weight != 0.0
    Hv[1:n] .= obj_weight * (nls.A' * (nls.A * v) )
  else
    Hv[1:n] .= 0.0
  end
  return Hv
end
