export AbstractBatchNLPModel
export obj!

"""
    AbstractBatchNLPModel

Abstract base type for batched nonlinear optimization models.

Each model in the batch has the same number of variables and constraints,
and the sparsity patterns of the Jacobian and the Hessian of the Lagrangian are identical across the batch.
"""
abstract type AbstractBatchNLPModel{T, S} end

function obj(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractMatrix) where {T, S}
  bf = S(undef, 1, bnlp.meta.nbatch) |> vec
  obj!(bnlp, bx, bf)
  return bf
end

function grad(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractMatrix) where {T, S}
  bg = S(undef, bnlp.meta.nvar, bnlp.meta.nbatch)
  grad!(bnlp, bx, bg)
  return bg
end

function cons(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractMatrix) where {T, S}
  bc = S(undef, bnlp.meta.ncon, bnlp.meta.nbatch)
  cons!(bnlp, bx, bc)
  return bc
end

function jac_structure(bnlp::AbstractBatchNLPModel)
  jrows = Vector{Int}(undef, bnlp.meta.nnzj)
  jcols = Vector{Int}(undef, bnlp.meta.nnzj)
  jac_structure!(bnlp, jrows, jcols)
  return (jrows, jcols)
end

function jac_coord(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractMatrix) where {T, S}
  bjvals = S(undef, bnlp.meta.nnzj, bnlp.meta.nbatch)
  jac_coord!(bnlp, bx, bjvals)
  return bjvals
end

function jprod(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractMatrix, bv::AbstractMatrix) where {T, S}
  bJv = S(undef, bnlp.meta.ncon, bnlp.meta.nbatch)
  jprod!(bnlp, bx, bv, bJv)
  return bJv
end

function jtprod(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractMatrix, bv::AbstractMatrix) where {T, S}
  bJtv = S(undef, bnlp.meta.nvar, bnlp.meta.nbatch)
  jtprod!(bnlp, bx, bv, bJtv)
  return bJtv
end

function hess_structure(bnlp::AbstractBatchNLPModel)
  hrows = Vector{Int}(undef, bnlp.meta.nnzh)
  hcols = Vector{Int}(undef, bnlp.meta.nnzh)
  hess_structure!(bnlp, hrows, hcols)
  return hrows, hcols
end

function hess_coord(
  bnlp::AbstractBatchNLPModel{T, S},
  bx::AbstractMatrix,
  by::AbstractMatrix,
  bobj_weight::AbstractVector,
) where {T, S}
  bhvals = S(undef, bnlp.meta.nnzh, bnlp.meta.nbatch)
  hess_coord!(bnlp, bx, by, bobj_weight, bhvals)
  return bhvals
end

function hprod(
  bnlp::AbstractBatchNLPModel{T, S},
  bx::AbstractMatrix,
  by::AbstractMatrix,
  bv::AbstractMatrix,
  bobj_weight::AbstractVector,
) where {T, S}
  bHv = S(undef, bnlp.meta.nvar, bnlp.meta.nbatch)
  hprod!(bnlp, bx, by, bv, bobj_weight, bHv)
  return bHv
end
