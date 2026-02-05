export AbstractBatchNLPModel

export batch_obj, batch_obj!
export batch_grad, batch_grad!
export batch_cons, batch_cons!
export batch_jac_structure, batch_jac_structure!
export batch_hess_structure, batch_hess_structure!
export batch_jac_coord, batch_jac_coord!
export batch_hess_coord, batch_hess_coord!
export batch_jprod, batch_jprod!
export batch_jtprod, batch_jtprod!
export batch_hprod, batch_hprod!

"""
    AbstractBatchNLPModel

Abstract base type for batched nonlinear optimization models.

Each model in the batch has the same number of variables and constraints,
and the sparsity patterns of the Jacobian and the Hessian of the Lagrangian are identical across the batch.
"""
abstract type AbstractBatchNLPModel{T, S} end

"""
    bf = batch_obj(bnlp, bx)
"""
function batch_obj(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractVector) where {T, S}
  @lencheck (bnlp.meta.nvar * bnlp.meta.nbatch) bx
  bf = S(undef, bnlp.meta.nbatch)
  batch_obj!(bnlp, bx, bf)
  return bf
end

"""
    bf = batch_obj!(bnlp, bx, bf)
"""
function batch_obj! end

"""
    bg = batch_grad(bnlp, bx)

This function is only available if `bnlp.meta.grad_available` is set to `true`.
"""
function batch_grad(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractVector) where {T, S}
  @lencheck (bnlp.meta.nvar * bnlp.meta.nbatch) bx
  bg = S(undef, bnlp.meta.nvar * bnlp.meta.nbatch)
  batch_grad!(bnlp, bx, bg)
  return bg
end

"""
    bg = batch_grad!(bnlp, bx, bg)

This function is only available if `bnlp.meta.grad_available` is set to `true`.
"""
function batch_grad! end

"""
    bc = batch_cons(bnlp, bx)
"""
function batch_cons(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractVector) where {T, S}
  @lencheck (bnlp.meta.nvar * bnlp.meta.nbatch) bx
  bc = S(undef, bnlp.meta.ncon * bnlp.meta.nbatch)
  batch_cons!(bnlp, bx, bc)
  return bc
end

"""
    bc = batch_cons!(bnlp, bx, bc)
"""
function batch_cons! end

"""
    (jrows, jcols) = batch_jac_structure(bnlp)

This function is only available if `bnlp.meta.jac_available` is set to `true`.
"""
function batch_jac_structure(bnlp::AbstractBatchNLPModel{T, S}) where {T, S}
  jrows = Vector{Int}(undef, bnlp.meta.nnzj)
  jcols = Vector{Int}(undef, bnlp.meta.nnzj)
  batch_jac_structure!(bnlp, jrows, jcols)
  return (jrows, jcols)
end

"""
    (jrows, jcols) = batch_jac_structure!(bnlp, jrows, jcols)

This function is only available if `bnlp.meta.jac_available` is set to `true`.
"""
function batch_jac_structure! end 

"""
    bjvals = batch_jac_coord(bnlp, bx)

This function is only available if `bnlp.meta.jac_available` is set to `true`.
"""
function batch_jac_coord(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractVector) where {T, S}
  @lencheck (bnlp.meta.nvar * bnlp.meta.nbatch) bx
  bjvals = S(undef, bnlp.meta.nnzj * bnlp.meta.nbatch)
  batch_jac_coord!(bnlp, bx, bjvals)
  return bjvals
end

"""
    bjvals = batch_jac_coord!(bnlp, bx, bjvals)

This function is only available if `bnlp.meta.jac_available` is set to `true`.
"""
function batch_jac_coord! end

"""
    bJv = batch_jprod(bnlp, bx, bv)

This function is only available if `bnlp.meta.jprod_available` is set to `true`.
"""
function batch_jprod(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractVector, bv::AbstractVector) where {T, S}
  @lencheck (bnlp.meta.nvar * bnlp.meta.nbatch) bx bv
  bJv = S(undef, bnlp.meta.ncon * bnlp.meta.nbatch)
  batch_jprod!(bnlp, bx, bv, bJv)
  return bJv
end

"""
    bJv = batch_jprod!(bnlp, bx, bv, bJv)

This function is only available if `bnlp.meta.jprod_available` is set to `true`.
"""
function batch_jprod! end

"""
    bJtv = batch_jtprod(bnlp, bx, bv)

This function is only available if `bnlp.meta.jtprod_available` is set to `true`.
"""
function batch_jtprod(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractVector, bv::AbstractVector) where {T, S}
  @lencheck (bnlp.meta.nvar * bnlp.meta.nbatch) bx
  @lencheck (bnlp.meta.ncon * bnlp.meta.nbatch) bv
  bJtv = S(undef, bnlp.meta.nvar * bnlp.meta.nbatch)
  batch_jtprod!(bnlp, bx, bv, bJtv)
  return bJtv
end

"""
    bJtv = batch_jtprod!(bnlp, bx, bv, bJtv)

This function is only available if `bnlp.meta.jtprod_available` is set to `true`.
"""
function batch_jtprod! end

"""
    (hrows, hcols) = batch_hess_structure(bnlp)

This function is only available if `bnlp.meta.hess_available` is set to `true`.
"""
function batch_hess_structure(bnlp::AbstractBatchNLPModel{T,S}) where {T, S}
  hrows = Vector{Int}(undef, bnlp.meta.nnzh)
  hcols = Vector{Int}(undef, bnlp.meta.nnzh)
  batch_hess_structure!(bnlp, hrows, hcols)
  return hrows, hcols
end

"""
    (hrows, hcols) = batch_hess_structure!(bnlp, hrows, hcols)

This function is only available if `bnlp.meta.hess_available` is set to `true`.
"""
function batch_hess_structure! end

"""
    bhvals = batch_hess_coord(bnlp, bx, by, bobj_weight)

This function is only available if `bnlp.meta.hess_available` is set to `true`.
"""
function batch_hess_coord(
  bnlp::AbstractBatchNLPModel{T, S},
  bx::AbstractVector,
  by::AbstractVector,
  bobj_weight::AbstractVector,
) where {T, S}
  @lencheck (bnlp.meta.nvar * bnlp.meta.nbatch) bx
  @lencheck (bnlp.meta.ncon * bnlp.meta.nbatch) by
  @lencheck bnlp.meta.nbatch bobj_weight
  bhvals = S(undef, bnlp.meta.nnzh * bnlp.meta.nbatch)
  return batch_hess_coord!(bnlp, bx, by, bobj_weight, bhvals)
end

"""
    bhvals = batch_hess_coord!(bnlp, bx, by, bobj_weight, bhvals)

This function is only available if `bnlp.meta.hess_available` is set to `true`.
"""
function batch_hess_coord! end

"""
    bHv = batch_hprod(bnlp, bx, by, bv, bobj_weight)

This function is only available if `bnlp.meta.hprod_available` is set to `true`.
"""
function batch_hprod(
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
  batch_hprod!(bnlp, bx, by, bv, bobj_weight, bHv)
  return bHv
end

"""
    bHv = batch_hprod!(bnlp, bx, by, bv, bobj_weight, bHv)

This function is only available if `bnlp.meta.hprod_available` is set to `true`.
"""
function batch_hprod! end
