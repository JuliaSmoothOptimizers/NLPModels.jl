export AbstractLinearLeastSquaresModel, LLSModel

abstract type AbstractLinearLeastSquaresModel <: AbstractNLPModel end

"""
    nlp = LLSModel(A, b; σls, lvar, uvar, C, lcon, ucon)

Creates a Linear Least Squares model σls*½‖Ax - b‖² with optional bounds
`lvar ≦ x ≦ uvar` and optional linear constraints `lcon ≦ Cx ≦ ucon`.
"""
mutable struct LLSModel <: AbstractLinearLeastSquaresModel
  meta :: NLPModelMeta
  counters :: Counters

  A :: Union{AbstractMatrix, AbstractLinearOperator}
  b :: AbstractVector
  σls :: Float64
  C :: Union{AbstractMatrix, AbstractLinearOperator}
end

function LLSModel(A :: Union{AbstractMatrix, LinearOperator}, b :: AbstractVector;
                  σls :: Float64 = 1.0,
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

  meta = NLPModelMeta(n, nobjs=0, nlsequ=0, llsrows=m,
                      x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, y0=y0, lin=1:ncon,
                      nln=Int[], lcon=lcon, ucon=ucon, nnzj=nnzj)

  return LLSModel(meta, Counters(), A, b, σls, C)
end

function residual!(nlp :: AbstractLinearLeastSquaresModel, x :: AbstractVector, Fx :: AbstractVector)
  increment!(nlp, :neval_residual)
  Fx .= nlp.A * x - nlp.b
  return Fx
end

function residual(nlp :: AbstractLinearLeastSquaresModel, x :: AbstractVector)
  increment!(nlp, :neval_residual)
  return nlp.A * x - nlp.b
end

function jac_residual(nlp :: AbstractLinearLeastSquaresModel, x :: AbstractVector)
  increment!(nlp, :neval_jac_residual)
  if isa(nlp.A, LinearOperator)
    error("Jacobian is a LinearOperator. Use `jac_op_residual` instead.")
  else
    return nlp.A
  end
end

function jprod_residual!(nlp :: AbstractLinearLeastSquaresModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jprod_residual)
  Jv .= nlp.A * v
  return Jv
end

function jprod_residual(nlp :: AbstractLinearLeastSquaresModel, x :: AbstractVector, v :: AbstractVector)
  increment!(nlp, :neval_jprod_residual)
  return nlp.A * v
end

function jtprod_residual!(nlp :: AbstractLinearLeastSquaresModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod_residual)
  Jtv .= nlp.A' * v
  return Jtv
end

function jtprod_residual(nlp :: AbstractLinearLeastSquaresModel, x :: AbstractVector, v :: AbstractVector)
  increment!(nlp, :neval_jtprod_residual)
  return nlp.A' * v
end

function jac_op_residual(nlp :: AbstractLinearLeastSquaresModel, x :: AbstractVector)
  if nlp.A isa AbstractLinearOperator
    return nlp.A
  else
    return LinearOperator(nlp.A)
  end
end

function jac_op_residual!(nlp :: AbstractLinearLeastSquaresModel, x :: AbstractVector,
                          Jv :: AbstractVector, Jtv :: AbstractVector)
  # Ignoring preallocation
  if nlp.A isa AbstractLinearOperator
    return nlp.A
  else
    return LinearOperator(nlp.A)
  end
end

function hess_residual(nlp :: AbstractLinearLeastSquaresModel, i :: Int, x :: AbstractVector)
  increment!(nlp, :neval_hess_residual)
  n = size(nlp.A, 2)
  return zeros(n, n)
end

function hprod_residual!(nlp :: AbstractLinearLeastSquaresModel, i :: Int, x :: AbstractVector, v :: AbstractVector, Hiv :: AbstractVector)
  increment!(nlp, :neval_hprod_residual)
  fill!(Hiv, 0.0)
  return Hiv
end

function cons(nlp :: LLSModel, x :: AbstractVector)
  increment!(nlp, :neval_cons)
  return nlp.C * x
end

function cons!(nlp :: LLSModel, x :: AbstractVector, c :: AbstractVector)
  increment!(nlp, :neval_cons)
  c[1:nlp.meta.ncon] = nlp.C * x
  return c
end

function jac_coord(nlp :: LLSModel, x :: AbstractVector)
  increment!(nlp, :neval_jac)
  if isa(nlp.C, LinearOperator)
    error("jac_coord is not defined for LinearOperators")
  end
  if isa(nlp.C, SparseMatrixCSC)
    return findnz(nlp.C)
  else
    I = findall(!iszero, nlp.C)
    return (getindex.(I, 1), getindex.(I, 2), nlp.C[I])
  end
end

function jac(nlp :: LLSModel, x :: AbstractVector)
  increment!(nlp, :neval_jac)
  return nlp.C
end

function jprod(nlp :: LLSModel, x :: AbstractVector, v :: AbstractVector)
  increment!(nlp, :neval_jprod)
  return nlp.C * v
end

function jprod!(nlp :: LLSModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jprod)
  Jv[1:nlp.meta.ncon] = nlp.C * v
  return Jv
end

function jtprod(nlp :: LLSModel, x :: AbstractVector, v :: AbstractVector)
  increment!(nlp, :neval_jtprod)
  return nlp.C' * v
end

function jtprod!(nlp :: LLSModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod)
  Jtv[1:nlp.meta.nvar] = nlp.C' * v
  return Jtv
end

function hess(nlp :: LLSModel, x :: AbstractVector; obj_weight = 1.0, y :: AbstractVector = Float64[])
  increment!(nlp, :neval_hess)
  if obj_weight != 0.0
    if isa(nlp.A, LinearOperator)
      error("hess is not defined for LinearOperators")
    end
    return tril(obj_weight * (nlp.A' * nlp.A))
  else
    n = length(x)
    return zeros(n, n)
  end
end

function hess_coord(nlp :: LLSModel, x :: AbstractVector; obj_weight = 1.0, y :: AbstractVector = Float64[])
  H = hess(nlp, x, obj_weight=obj_weight, y=y)
  if isa(H, SparseMatrixCSC)
    return findnz(H)
  else
    I = findall(!iszero, H)
    return (getindex.(I, 1), getindex.(I, 2), H[I])
  end
end

function hprod(nlp :: LLSModel, x :: AbstractVector, v :: AbstractVector;
    obj_weight = 1.0, y :: AbstractVector = Float64[])
  Hv = zeros(nlp.meta.nvar)
  return hprod!(nlp, x, v, Hv, obj_weight=obj_weight, y=y)
end

function hprod!(nlp :: LLSModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector;
    obj_weight = 1.0, y :: AbstractVector = Float64[])
  increment!(nlp, :neval_hprod)
  n = length(x)
  if obj_weight != 0.0
    Hv[1:n] .= obj_weight * (nlp.A' * (nlp.A * v) )
  else
    Hv[1:n] .= 0.0
  end
  return Hv
end
