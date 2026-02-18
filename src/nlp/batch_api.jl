export AbstractBatchNLPModel

"""
    AbstractBatchNLPModel

Abstract base type for batched nonlinear optimization models.

Each model in the batch has the same number of variables and constraints,
and the sparsity patterns of the Jacobian and the Hessian of the Lagrangian are identical across the batch.
"""
abstract type AbstractBatchNLPModel{T, S} end

"""
    bf = obj(bnlp, bx)
"""
function obj(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractVector) where {T, S}
  @lencheck (bnlp.meta.nvar * bnlp.meta.nbatch) bx
  bf = S(undef, bnlp.meta.nbatch)
  obj!(bnlp, bx, bf)
  return bf
end

"""
    bf = obj!(bnlp, bx, bf)
"""
function obj! end

"""
    bg = grad(bnlp, bx)

This function is only available if `bnlp.meta.grad_available` is set to `true`.
"""
function grad(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractVector) where {T, S}
  @lencheck (bnlp.meta.nvar * bnlp.meta.nbatch) bx
  bg = S(undef, bnlp.meta.nvar * bnlp.meta.nbatch)
  grad!(bnlp, bx, bg)
  return bg
end

"""
    bg = grad!(bnlp, bx, bg)

This function is only available if `bnlp.meta.grad_available` is set to `true`.
"""
# function grad! end

"""
    bc = cons(bnlp, bx)
"""
function cons(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractVector) where {T, S}
  @lencheck (bnlp.meta.nvar * bnlp.meta.nbatch) bx
  bc = S(undef, bnlp.meta.ncon * bnlp.meta.nbatch)
  cons!(bnlp, bx, bc)
  return bc
end

"""
    bc = cons!(bnlp, bx, bc)
"""
# function cons! end

"""
    (jrows, jcols) = jac_structure(bnlp)

This function is only available if `bnlp.meta.jac_available` is set to `true`.
"""
function jac_structure(bnlp::AbstractBatchNLPModel{T, S}) where {T, S}
  jrows = Vector{Int}(undef, bnlp.meta.nnzj)
  jcols = Vector{Int}(undef, bnlp.meta.nnzj)
  jac_structure!(bnlp, jrows, jcols)
  return (jrows, jcols)
end

"""
    (jrows, jcols) = jac_structure!(bnlp, jrows, jcols)

This function is only available if `bnlp.meta.jac_available` is set to `true`.
"""
# function jac_structure! end

"""
    bjvals = jac_coord(bnlp, bx)

This function is only available if `bnlp.meta.jac_available` is set to `true`.
"""
function jac_coord(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractVector) where {T, S}
  @lencheck (bnlp.meta.nvar * bnlp.meta.nbatch) bx
  bjvals = S(undef, bnlp.meta.nnzj * bnlp.meta.nbatch)
  jac_coord!(bnlp, bx, bjvals)
  return bjvals
end

"""
    bjvals = jac_coord!(bnlp, bx, bjvals)

This function is only available if `bnlp.meta.jac_available` is set to `true`.
"""
# function jac_coord! end

"""
    bJx = jac_dense!(bnlp, bx, bJx)

This function is only available if `bnlp.meta.jac_available` is set to `true`.
"""
# function jac_dense! end

"""
    bJv = jprod(bnlp, bx, bv)

This function is only available if `bnlp.meta.jprod_available` is set to `true`.
"""
function jprod(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractVector, bv::AbstractVector) where {T, S}
  @lencheck (bnlp.meta.nvar * bnlp.meta.nbatch) bx bv
  bJv = S(undef, bnlp.meta.ncon * bnlp.meta.nbatch)
  jprod!(bnlp, bx, bv, bJv)
  return bJv
end

"""
    bJv = jprod!(bnlp, bx, bv, bJv)

This function is only available if `bnlp.meta.jprod_available` is set to `true`.
"""
# function jprod! end

"""
    bJtv = jtprod(bnlp, bx, bv)

This function is only available if `bnlp.meta.jtprod_available` is set to `true`.
"""
function jtprod(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractVector, bv::AbstractVector) where {T, S}
  @lencheck (bnlp.meta.nvar * bnlp.meta.nbatch) bx
  @lencheck (bnlp.meta.ncon * bnlp.meta.nbatch) bv
  bJtv = S(undef, bnlp.meta.nvar * bnlp.meta.nbatch)
  jtprod!(bnlp, bx, bv, bJtv)
  return bJtv
end

"""
    bJtv = jtprod!(bnlp, bx, bv, bJtv)

This function is only available if `bnlp.meta.jtprod_available` is set to `true`.
"""
# function jtprod! end

"""
    (hrows, hcols) = hess_structure(bnlp)

This function is only available if `bnlp.meta.hess_available` is set to `true`.
"""
function hess_structure(bnlp::AbstractBatchNLPModel{T,S}) where {T, S}
  hrows = Vector{Int}(undef, bnlp.meta.nnzh)
  hcols = Vector{Int}(undef, bnlp.meta.nnzh)
  hess_structure!(bnlp, hrows, hcols)
  return hrows, hcols
end

"""
    (hrows, hcols) = hess_structure!(bnlp, hrows, hcols)

This function is only available if `bnlp.meta.hess_available` is set to `true`.
"""
# function hess_structure! end

"""
    bhvals = hess_coord(bnlp, bx, by, bobj_weight)

This function is only available if `bnlp.meta.hess_available` is set to `true`.
"""
function hess_coord(
  bnlp::AbstractBatchNLPModel{T, S},
  bx::AbstractVector,
  by::AbstractVector,
  bobj_weight::AbstractVector,
) where {T, S}
  @lencheck (bnlp.meta.nvar * bnlp.meta.nbatch) bx
  @lencheck (bnlp.meta.ncon * bnlp.meta.nbatch) by
  @lencheck bnlp.meta.nbatch bobj_weight
  bhvals = S(undef, bnlp.meta.nnzh * bnlp.meta.nbatch)
  return hess_coord!(bnlp, bx, by, bobj_weight, bhvals)
end

"""
    bhvals = hess_coord!(bnlp, bx, by, bobj_weight, bhvals)

This function is only available if `bnlp.meta.hess_available` is set to `true`.
"""
# function hess_coord! end

"""
    bHx = hess_dense!(bnlp, bx, by, bHx, bobj_weight)

This function is only available when `bnlp.meta.hess_available` is set to `true`.
"""
# function hess_dense! end

"""
    bHv = hprod(bnlp, bx, by, bv, bobj_weight)

This function is only available if `bnlp.meta.hprod_available` is set to `true`.
"""
function hprod(
  bnlp::AbstractBatchNLPModel{T, S},
  bx::AbstractVector,
  by::AbstractVector,
  bv::AbstractVector,
  bobj_weight::AbstractVector,
) where {T, S}
  @lencheck (bnlp.meta.nvar * bnlp.meta.nbatch) bx bv
  @lencheck (bnlp.meta.ncon * bnlp.meta.nbatch) by
  @lencheck bnlp.meta.nbatch bobj_weight
  bHv = S(undef, bnlp.meta.nvar * bnlp.meta.nbatch)
  hprod!(bnlp, bx, by, bv, bobj_weight, bHv)
  return bHv
end

"""
    bHv = hprod!(bnlp, bx, by, bv, bobj_weight, bHv)

This function is only available if `bnlp.meta.hprod_available` is set to `true`.
"""
# function hprod! end
