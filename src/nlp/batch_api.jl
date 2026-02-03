export AbstractBatchNLPModel

export batch_obj, batch_obj!
export batch_grad, batch_grad!
export batch_cons, batch_cons!
export batch_jac_structure!, batch_jac_structure
export batch_hess_structure!, batch_hess_structure
export batch_jac_coord!, batch_jac_coord
export batch_hess_coord!, batch_hess_coord
export batch_jprod, batch_jprod!
export batch_jtprod, batch_jtprod!
export batch_hprod, batch_hprod!

"""
    AbstractBatchNLPModel

Abstract base type for batched nonlinear optimization models.

Each model in the batch has the same number of variables and constraints,
and the sparsity patterns of the Jacobian and the Hessian of the Lagrangian are identical across the batch.
"""
abstract type AbstractBatchNLPModel{T,S,VI} end

"""
    bf = batch_obj(bnlp, bx)

The input `bx` contains `bnlp.nbatch` points stacked in a full strided (column-wise) layout.
The output `bf` stores the objective value for each batch entry.
"""
function batch_obj(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractVector) where {T, S}
  @lencheck (bnlp.meta.nvar * bnlp.meta.nbatch) bx
  bf = S(undef, bnlp.meta.nbatch)
  batch_obj!(bnlp, bx, bf)
  return bf
end

"""
    bf = batch_obj!(bnlp, bx, bf)

Evaluate ``[ f(x₁) | f(x₂) | ... ]``, the objective function of `bnlp` at `bx = [ x₁ | x₂ | ... ]` in place.
Both input `bx` and output `bf` follow a full strided (column-wise) batch layout.
"""
function batch_obj! end

"""
    bg = batch_grad(bnlp, bx)

Evaluate ``∇f(x)``, the gradient of the objective function at `bx`.
The gradients are returned in a full strided (column-wise) layout, with one gradient vector per batch entry.
This function is only available if `bnlp.meta.grad_available` is set to `true`.
"""
function batch_grad(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractVector) where {T, S}
  @lencheck (bnlp.meta.nvar * bnlp.meta.nbatch) bx
  bg = S(undef, bnlp.meta.nvar * bnlp.meta.nbatch)
  batch_grad!(nlp, bx, bg)
  return bg
end

"""
    bg = batch_grad!(bnlp, bx, bg)

Evaluate ``∇f(bx)`` for a batch of points stored in `bx` in place.
Both input `bx` and output `bg` use a full strided (column-wise) batch layout.
This function is only available if `bbnlp.meta.grad_available` is set to `true`.
"""
function batch_grad! end

"""
    bc = batch_cons(bnlp, bx)

Evaluate ``c(x)``, the constraints at `x`.
The constraints are returned in a full strided (column-wise) layout, with one constraint
vector per batch entry.
"""
function batch_cons(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractVector) where {T, S}
  @lencheck (bnlp.meta.nvar * bnlp.meta.nbatch) bx
  bc = S(undef, bnlp.meta.ncon * bnlp.meta.nbatch)
  batch_cons!(bnlp, bx, bc)
  return bc
end

"""
    bc = batch_cons!(bnlp, bx, bc)

Evaluate ``c(x)``, the constraints at `x` in place.
Both input `bx` and output `c` follow a full strided (column-wise) batch layout.
"""
function batch_cons! end

"""
    (jrows, jcols) = batch_jac_structure(bnlp)

Return the structure of the constraints Jacobian in sparse coordinate format.
This function is only available if `bbnlp.meta.jac_available` is set to `true`.
"""
function batch_jac_structure(bnlp::AbstractBatchNLPModel{T, S, VI}) where {T, S, VI}
  jrows = VI(undef, bnlp.meta.nnzj)
  jcols = VI(undef, bnlp.meta.nnzj)
  batch_jac_structure!(bnlp, jrows, jcols)
  return (jrows, jcols)
end

"""
    (jrows, jcols) = batch_jac_structure!(bnlp, jrows, jcols)

Return the structure of the constraints Jacobian in sparse coordinate format in place.
This function is only available if `bbnlp.meta.jac_available` is set to `true`.
"""
function batch_jac_structure! end 

"""
    bjvals = batch_jac_coord(bnlp, bx)

Evaluate ``J(x)``, the constraints Jacobian at `x` in sparse coordinate format.
This function is only available if `bbnlp.meta.jac_available` is set to `true`.
"""
function batch_jac_coord(bnlp::AbstractBatchNLPModel, bx::AbstractVector, bjvals::AbstractVector)
  @lencheck (bnlp.meta.nvar * bnlp.meta.nbatch) bx
  @lencheck (bnlp.meta.nnzj * bnlp.meta.nbatch) bjvals
  batch_jac_coord!(bnlp, bx, bjbvals)
  return bjvals
end

"""
    bjvals = batch_jac_coord!(bnlp, bx, bjvals)

Evaluate ``J(x)``, the constraints Jacobian at `x` in sparse coordinate format, rewriting `vals`.
This function is only available if `bbnlp.meta.jac_available` is set to `true`.
"""
function batch_jac_coord! end

"""
    bJv = batch_jprod(bnlp, bx, bv)

Evaluate ``J(x)v``, the Jacobian-vector product at `x`.
This function is only available if `bbnlp.meta.jprod_available` is set to `true`.
"""
function batch_jprod(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractVector, bv::AbstractVector) where {T, S}
  @lencheck (bnlp.meta.nvar * bnlp.meta.nbatch) bx bv
  bJv = S(undef, bnlp.meta.ncon * bnlp.meta.nbatch)
  batch_jprod!(bnlp, bx, bv, bJv)
  return bJv
end

"""
    bJv = batch_jprod!(bnlp, bx, bv, bJv)

Evaluate ``J(x)v``, the Jacobian-vector product at `x` in place.
This function is only available if `bbnlp.meta.jprod_available` is set to `true`.
"""
function batch_jprod! end

"""
    bJtv = batch_jtprod(bnlp, bx, bv)

Evaluate ``J(x)^Tv``, the transposed-Jacobian-vector product at `x`.
This function is only available if `bbnlp.meta.jtprod_available` is set to `true`.
"""
function batch_jtprod(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractVector, bv::AbstractVector) where {T, S}
  @lencheck (bnlp.meta.nvar * bnlp.meta.nbatch) bx
  @lencheck (bnlp.meta.ncon * bnlp.meta.nbatch) bv
  bJtv = S(undef, bnlp.meta.nvar * bnlp.meta.nbatch)
  batch_jtprod!(bnlp, bx, bv, bJtv)
  return bJtv
end

"""
    bJtv = batch_jtprod!(bnlp, bx, bv, Jtv)

Evaluate ``J(x)^Tv``, the transposed-Jacobian-vector product at `x` in place.
If the problem has linear and nonlinear constraints, this function allocates.
This function is only available if `bbnlp.meta.jtprod_available` is set to `true`.
"""
function batch_jtprod! end

"""
    (hrows, hcols) = batch_hess_structure(bnlp)

Return the structure of the Lagrangian Hessian in sparse coordinate format.
This function is only available if `bbnlp.meta.hess_available` is set to `true`.
"""
function batch_hess_structure(bnlp::AbstractBatchNLPModel{T,S,VI}) where {T, S, VI}
  hrows = VI(undef, bnlp.meta.nnzh)
  hcols = VI(undef, bnlp.meta.nnzh)
  batch_hess_structure!(bnlp, hrows, hcols)
  return hrows, hcols
end

"""
    (hrows, hcols) = batch_hess_structure!(bnlp, hrows, hcols)

Return the structure of the Lagrangian Hessian in sparse coordinate format in place.
This function is only available if `bbnlp.meta.hess_available` is set to `true`.
"""
function batch_hess_structure! end

"""
    bhvals = batch_hess_coord(bnlp, bx, by, bobj_weight)

Evaluate the Lagrangian Hessian at `(x,y)` in sparse coordinate format,
with objective function scaled by `obj_weight`, i.e.,
$(LAGRANGIAN_HESSIAN).
Only the lower triangle is returned.
This function is only available if `nlp.meta.hess_available` is set to `true`.
"""
function batch_hess_coord(
  bnlp::AbstractNLPModel{T, S},
  bx::AbstractVector,
  by::AbstractVector,
  bobj_weight::AbstractVector,
) where {T, S}
  @lencheck (bnlp.meta.nvar * bnlp.meta.nbatch) bx
  @lencheck (bnlp.meta.ncon * bnlp.meta.nbatch) by
  @lencheck bnlp.meta.nbatch bobj_weight
  vals = S(undef, nlp.meta.nnzh * bnlp.meta.nbatch)
  return batch_hess_coord!(bnlp, bx, by, bobj_weight, bhvals)
end

"""
    bhvals = batch_hess_coord!(bnlp, bx, by, bobj_weight, bhvals)

Evaluate the Lagrangian Hessian at `(x,y)` in sparse coordinate format,
with objective function scaled by `obj_weight`, i.e.,
$(LAGRANGIAN_HESSIAN), overwriting `vals`.
Only the lower triangle is returned.
This function is only available if `bbnlp.meta.hess_available` is set to `true`.
"""
function batch_hess_coord! end

"""
    bHv = batch_hprod(bnlp, bx, by, bv, bobj_weight)

Evaluate the product of the objective Hessian at `x` with the vector `v`,
with objective function scaled by `obj_weight`, where the objective Hessian is
$(OBJECTIVE_HESSIAN).
This function is only available if `bbnlp.meta.hprod_available` is set to `true`.
"""
function batch_hprod(
  bnlp::AbstractBatchNLPModel{T, S},
  bx::AbstractVector,
  by::AbstractVector,
  bv::AbstractVector,
  bobj_weight::AbstractVector,
) where {T, S}
  @lencheck (bnlp.meta.nvar * bnlp.meta.nbatch) bx bv
  @lencheck bnlp.meta.nbatch bobj_weight
  bHv = S(undef, bnlp.meta.nvar * bnlp.meta.nbatch)
  batch_hprod!(bnlp, bx, bv, bHv, bobj_weight)
  return bHv
end

"""
    bHv = batch_hprod!(bnlp, bx, by, bv, bobj_weight, bHv)

Evaluate the product of the objective Hessian at `x` with the vector `v` in
place, with objective function scaled by `obj_weight`, where the objective Hessian is
$(OBJECTIVE_HESSIAN).
This function is only available if `bbnlp.meta.hprod_available` is set to `true`.
"""
function batch_hprod! end
