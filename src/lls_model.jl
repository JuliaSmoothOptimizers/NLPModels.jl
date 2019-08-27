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

  A :: Union{AbstractMatrix, AbstractLinearOperator}
  b :: AbstractVector
  C :: Union{AbstractMatrix, AbstractLinearOperator}
end

function LLSModel(A :: Union{AbstractMatrix, AbstractLinearOperator}, b :: AbstractVector;
                  x0 :: AbstractVector = zeros(size(A,2)),
                  lvar :: AbstractVector = fill(-Inf, size(A, 2)),
                  uvar :: AbstractVector = fill(Inf, size(A, 2)),
                  C :: Union{AbstractMatrix, AbstractLinearOperator} = Matrix{Float64}(undef, 0, 0),
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
  nnzjF, nnzh = if A isa AbstractSparseMatrix
    nnz(A), nnz(tril(A' * A))
  elseif A isa AbstractLinearOperator
    0, 0 # Can't provide coordinates
  else
    m * n, div(n * (n + 1), 2) # Full
  end
  nnzj = if ncon == 0 || C isa AbstractLinearOperator
    0
  elseif C isa AbstractSparseMatrix
    nnz(C)
  else
    n * ncon
  end

  meta = NLPModelMeta(n, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, y0=y0, lin=1:ncon,
                      nln=Int[], lcon=lcon, ucon=ucon, nnzj=nnzj, nnzh=nnzh)

  nls_meta = NLSMeta(m, n, nnzj=nnzjF, nnzh=0)

  return LLSModel(meta, nls_meta, NLSCounters(), A, b, C)
end

function residual!(nls :: LLSModel, x :: AbstractVector, Fx :: AbstractVector)
  increment!(nls, :neval_residual)
  Fx[:] = nls.A * x - nls.b
  return Fx
end

function jac_residual(nls :: LLSModel, x :: AbstractVector)
  increment!(nls, :neval_jac_residual)
  if isa(nls.A, AbstractLinearOperator)
    error("Jacobian is a LinearOperator. Use `jac_op_residual` instead.")
  else
    return nls.A
  end
end

function jac_structure_residual!(nls :: LLSModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  if isa(nls.A, AbstractLinearOperator)
    error("Jacobian is a LinearOperator. Use `jac_op_residual` instead.")
  elseif nls.A isa AbstractSparseMatrix
    fnz = findnz(nls.A)
    rows .= fnz[1]
    cols .= fnz[2]
    return rows, cols
  else
    m, n = size(nls.A)
    I = ((i,j) for i = 1:m, j = 1:n)
    rows .= getindex.(I, 1)[:]
    cols .= getindex.(I, 2)[:]
    return rows, cols
  end
end

function jac_coord_residual!(nls :: LLSModel, x :: AbstractVector, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer}, vals :: AbstractVector)
  increment!(nls, :neval_jac_residual)
  if isa(nls.A, AbstractLinearOperator)
    error("Jacobian is a LinearOperator. Use `jac_op_residual` instead.")
  else
    for k = 1:nls.nls_meta.nnzj
      i, j = rows[k], cols[k]
      vals[k] = nls.A[i,j]
    end
  end
  return (rows,cols,vals)
end

function jac_coord_residual(nls :: LLSModel, x :: AbstractVector)
  increment!(nls, :neval_jac_residual)
  if isa(nls.A, AbstractLinearOperator)
    error("Jacobian is a LinearOperator. Use `jac_op_residual` instead.")
  elseif nls.A isa AbstractSparseMatrix
    return findnz(nls.A)
  else
    m, n = size(nls.A)
    I = ((i,j) for i = 1:m, j = 1:n)
    return (getindex.(I, 1)[:], getindex.(I, 2)[:], nls.A[:])
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

function hess_residual(nls :: LLSModel, x :: AbstractVector, v :: AbstractVector)
  increment!(nls, :neval_hess_residual)
  n = size(nls.A, 2)
  return zeros(n, n)
end

function hess_structure_residual!(nls :: LLSModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  return rows, cols
end

function hess_coord_residual!(nls :: LLSModel, x :: AbstractVector, v :: AbstractVector, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer}, vals :: AbstractVector)
  increment!(nls, :neval_hess_residual)
  return (rows, cols, vals)
end

function hess_coord_residual(nls :: LLSModel, x :: AbstractVector, v :: AbstractVector)
  increment!(nls, :neval_hess_residual)
  return (Int[], Int[], Float64[])
end

function jth_hess_residual(nls :: LLSModel, x :: AbstractVector, i :: Int)
  increment!(nls, :neval_jhess_residual)
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

function jac_structure!(nls :: LLSModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  if isa(nls.C, AbstractLinearOperator)
    error("jac_coord is not defined for LinearOperators")
  end
  if isa(nls.C, AbstractSparseMatrix)
    fnz = findnz(nls.C)
    rows .= fnz[1]
    cols .= fnz[2]
    return rows, cols
  else
    m, n = size(nls.C)
    I = ((i,j) for i = 1:m, j = 1:n)
    rows .= getindex.(I, 1)[:]
    cols .= getindex.(I, 2)[:]
    return rows, cols
  end
end

function jac_coord!(nls :: LLSModel, x :: AbstractVector, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer}, vals :: AbstractVector)
  increment!(nls, :neval_jac)
  if isa(nls.C, AbstractLinearOperator)
    error("jac_coord is not defined for LinearOperators")
  end
  for k = 1:nls.meta.nnzj
    i, j = rows[k], cols[k]
    vals[k] = nls.C[i,j]
  end
  return (rows, cols, vals)
end

function jac_coord(nls :: LLSModel, x :: AbstractVector)
  increment!(nls, :neval_jac)
  if isa(nls.C, AbstractLinearOperator)
    error("jac_coord is not defined for LinearOperators")
  end
  if isa(nls.C, AbstractSparseMatrix)
    return findnz(nls.C)
  else
    m, n = size(nls.C)
    I = ((i,j) for i = 1:m, j = 1:n)
    return (getindex.(I, 1)[:], getindex.(I, 2)[:], nls.C[:])
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
    if isa(nls.A, AbstractLinearOperator)
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
