export AbstractBatchNLPModel
export obj!

"""
    AbstractBatchNLPModel

Abstract base type for batched nonlinear optimization models.

Each model in the batch has the same number of variables and constraints,
and the sparsity patterns of the Jacobian and the Hessian of the Lagrangian are identical across the batch.
"""
abstract type AbstractBatchNLPModel{T, S} end

"""
    bf = obj(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix)
"""
function obj(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractMatrix) where {T, S}
  bf = S(undef, 1, bnlp.meta.nbatch) |> vec
  obj!(bnlp, bx, bf)
  return bf
end

"""
    bf = obj!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bf::AbstractVector)
"""
function obj! end

"""
    bg = grad(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix)

This function is only available if `bnlp.meta.grad_available` is set to `true`.
"""
function grad(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractMatrix) where {T, S}
  bg = S(undef, bnlp.meta.nvar, bnlp.meta.nbatch)
  grad!(bnlp, bx, bg)
  return bg
end

"""
    bg = grad!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bg::AbstractMatrix)

This function is only available if `bnlp.meta.grad_available` is set to `true`.
"""
# function grad! end

"""
    bc = cons(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix)
"""
function cons(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractMatrix) where {T, S}
  bc = S(undef, bnlp.meta.ncon, bnlp.meta.nbatch)
  cons!(bnlp, bx, bc)
  return bc
end

"""
    bc = cons!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bc::AbstractMatrix)
"""
# function cons! end

"""
    (jrows, jcols) = jac_structure(bnlp::AbstractBatchNLPModel)

This function is only available if `bnlp.meta.jac_available` is set to `true`.
"""
function jac_structure(bnlp::AbstractBatchNLPModel)
  jrows = Vector{Int}(undef, bnlp.meta.nnzj)
  jcols = Vector{Int}(undef, bnlp.meta.nnzj)
  jac_structure!(bnlp, jrows, jcols)
  return (jrows, jcols)
end

"""
    (jrows, jcols) = jac_structure!(bnlp::AbstractBatchNLPModel, jrows::AbstractVector, jcols::AbstractVector)

This function is only available if `bnlp.meta.jac_available` is set to `true`.
"""
# function jac_structure! end

"""
    bjvals = jac_coord(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix)

This function is only available if `bnlp.meta.jac_available` is set to `true`.
"""
function jac_coord(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractMatrix) where {T, S}
  bjvals = S(undef, bnlp.meta.nnzj, bnlp.meta.nbatch)
  jac_coord!(bnlp, bx, bjvals)
  return bjvals
end

"""
    bjvals = jac_coord!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bjvals::AbstractMatrix)

This function is only available if `bnlp.meta.jac_available` is set to `true`.
"""
# function jac_coord! end

"""
    bJx = jac_dense!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bJx::AbstractArray)

This function is only available if `bnlp.meta.jac_available` is set to `true`.
"""
# function jac_dense! end

"""
    bJv = jprod(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bv::AbstractMatrix)

This function is only available if `bnlp.meta.jprod_available` is set to `true`.
"""
function jprod(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractMatrix, bv::AbstractMatrix) where {T, S}
  bJv = S(undef, bnlp.meta.ncon, bnlp.meta.nbatch)
  jprod!(bnlp, bx, bv, bJv)
  return bJv
end

"""
    bJv = jprod!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bv::AbstractMatrix, bJv::AbstractMatrix)

This function is only available if `bnlp.meta.jprod_available` is set to `true`.
"""
# function jprod! end

"""
    bJtv = jtprod(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bv::AbstractMatrix)

This function is only available if `bnlp.meta.jtprod_available` is set to `true`.
"""
function jtprod(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractMatrix, bv::AbstractMatrix) where {T, S}
  bJtv = S(undef, bnlp.meta.nvar, bnlp.meta.nbatch)
  jtprod!(bnlp, bx, bv, bJtv)
  return bJtv
end

"""
    bJtv = jtprod!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bv::AbstractMatrix, bJtv::AbstractMatrix)

This function is only available if `bnlp.meta.jtprod_available` is set to `true`.
"""
# function jtprod! end

"""
    (hrows, hcols) = hess_structure(bnlp::AbstractBatchNLPModel)

This function is only available if `bnlp.meta.hess_available` is set to `true`.
"""
function hess_structure(bnlp::AbstractBatchNLPModel)
  hrows = Vector{Int}(undef, bnlp.meta.nnzh)
  hcols = Vector{Int}(undef, bnlp.meta.nnzh)
  hess_structure!(bnlp, hrows, hcols)
  return hrows, hcols
end

"""
    (hrows, hcols) = hess_structure!(bnlp::AbstractBatchNLPModel, hrows::AbstractVector, hcols::AbstractVector)

This function is only available if `bnlp.meta.hess_available` is set to `true`.
"""
# function hess_structure! end

"""
    bhvals = hess_coord(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, by::AbstractMatrix, bobj_weight::AbstractVector)

This function is only available if `bnlp.meta.hess_available` is set to `true`.
"""
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

"""
    bhvals = hess_coord!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, by::AbstractMatrix, bobj_weight::AbstractVector, bhvals::AbstractMatrix)

This function is only available if `bnlp.meta.hess_available` is set to `true`.
"""
# function hess_coord! end

"""
    bHx = hess_dense!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, by::AbstractMatrix, bobj_weight::AbstractVector, bHx::AbstractArray)

This function is only available when `bnlp.meta.hess_available` is set to `true`.
"""
# function hess_dense! end

"""
    bHv = hprod(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, by::AbstractMatrix, bv::AbstractMatrix, bobj_weight::AbstractVector)

This function is only available if `bnlp.meta.hprod_available` is set to `true`.
"""
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

"""
    bHv = hprod!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, by::AbstractMatrix, bv::AbstractMatrix, bobj_weight::AbstractVector, bHv::AbstractMatrix)

This function is only available if `bnlp.meta.hprod_available` is set to `true`.
"""
# function hprod! end
