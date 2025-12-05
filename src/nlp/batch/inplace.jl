export InplaceBatchNLPModel
struct InplaceBatchNLPModel{M} <: AbstractBatchNLPModel
  base_model::M
  updates
  counters::Counters
  batch_size::Int
end
function InplaceBatchNLPModel(base_model::M, updates) where {M}
  isempty(updates) && error("Cannot create InplaceBatchNLPModel from empty collection.")
  InplaceBatchNLPModel{M}(base_model, updates, Counters(), length(updates))
end

const INPLACE_OPERATOR_ERROR =
  "InplaceBatchNLPModel cannot return reusable linear operators because the base model is mutated per batch entry. Use ForEachBatchNLPModel instead."
_inplace_operator_error() = error(INPLACE_OPERATOR_ERROR)

# TODO: counters?
Base.length(vnlp::InplaceBatchNLPModel) = vnlp.batch_size


function _batch_map(f::F, bnlp::InplaceBatchNLPModel, xs::Vararg{T,N}) where {F,T,N}
  n = bnlp.batch_size
  @lencheck_tup n xs
  results = []
  resize!(results, n)
  for i = 1:n
    args_i = (x[i] for x in xs)
    bnlp.updates[i](bnlp.base_model)  # call update function
    results[i] = f(bnlp.base_model, args_i...)
  end
  return results
end

function _batch_map!(f::F, bnlp::InplaceBatchNLPModel, xs::Vararg{T,N}) where {F,T,N}
  n = bnlp.batch_size
  length(xs) == 0 && error("Cannot call _batch_map! without providing arguments.")
  @lencheck_tup n xs
  outputs = xs[end]
  inputs = length(xs) == 1 ? () : Base.ntuple(i -> xs[i], length(xs) - 1)
  @lencheck n outputs
  for i = 1:n
    args_i = (x[i] for x in inputs)
    bnlp.updates[i](bnlp.base_model)  # call update function
    f(bnlp.base_model, args_i..., outputs[i])
  end
  return outputs
end

function _batch_map_weight(f::F, bnlp::InplaceBatchNLPModel, obj_weights, xs::Vararg{T,N}) where {F,T,N}
  n = bnlp.batch_size
  @lencheck_tup n xs
  @lencheck n obj_weights
  results = []
  resize!(results, n)
  for i = 1:n
    args_i = (x[i] for x in xs)
    bnlp.updates[i](bnlp.base_model)  # call update function
    results[i] = f(bnlp.base_model, args_i...; obj_weight = obj_weights[i])
  end
  return results
end

function _batch_map_weight!(f::F, bnlp::InplaceBatchNLPModel, obj_weights, xs::Vararg{T,N}) where {F,T,N}
  n = bnlp.batch_size
  length(xs) == 0 && error("_batch_map_weight! with zero args")
  @lencheck_tup n xs
  @lencheck n obj_weights
  outputs = xs[end]
  inputs = length(xs) == 1 ? () : Base.ntuple(i -> xs[i], length(xs) - 1)
  @lencheck n outputs
  for i = 1:n
    args_i = (x[i] for x in inputs)
    bnlp.updates[i](bnlp.base_model)  # call update function
    f(bnlp.base_model, args_i..., outputs[i]; obj_weight = obj_weights[i])
  end
  return outputs
end

function _batch_map_tuple(f::F, bnlp::InplaceBatchNLPModel, xs::Vararg{T,N}) where {F,T,N}
  n = bnlp.batch_size
  @lencheck_tup n xs
  results = _batch_map(f, bnlp, xs...)

  first_result = first(results)
  T1, T2 = typeof(first_result[1]), typeof(first_result[2])
  vec1, vec2 = Vector{T1}(undef, n), Vector{T2}(undef, n)
  for i = 1:n
    vec1[i], vec2[i] = results[i]
  end
  return vec1, vec2
end

function _batch_map_tuple!(f::F, bnlp::InplaceBatchNLPModel, outputs, xs::Vararg{T,N}) where {F,T,N}
  n = bnlp.batch_size
  @lencheck_tup n xs
  @lencheck n outputs
  firsts = []
  resize!(firsts, n)
  for i = 1:n
    args_i = (x[i] for x in xs)
    bnlp.updates[i](bnlp.base_model)  # call update function
    firsts[i], _ = f(bnlp.base_model, args_i..., outputs[i])
  end
  return firsts, outputs
end

for fun in fieldnames(Counters)
  @eval function NLPModels.increment!(bnlp::InplaceBatchNLPModel, ::Val{$(Meta.quot(fun))})
    # sub-model counters are already incremented since we call their methods
    bnlp.counters.$fun += 1
  end
end


batch_jac_structure(bnlp::InplaceBatchNLPModel) =
  _batch_map(jac_structure, bnlp)
batch_jac_lin_structure(bnlp::InplaceBatchNLPModel) =
  _batch_map(jac_lin_structure, bnlp)
batch_jac_nln_structure(bnlp::InplaceBatchNLPModel) =
  _batch_map(jac_nln_structure, bnlp)
batch_hess_structure(bnlp::InplaceBatchNLPModel) =
  _batch_map(hess_structure, bnlp)
batch_obj(bnlp::InplaceBatchNLPModel, xs) =
  _batch_map(obj, bnlp, xs)
batch_grad(bnlp::InplaceBatchNLPModel, xs) =
  _batch_map(grad, bnlp, xs)
batch_cons(bnlp::InplaceBatchNLPModel, xs) =
  _batch_map(cons, bnlp, xs)
batch_cons_lin(bnlp::InplaceBatchNLPModel, xs) =
  _batch_map(cons_lin, bnlp, xs)
batch_cons_nln(bnlp::InplaceBatchNLPModel, xs) =
  _batch_map(cons_nln, bnlp, xs)
batch_jac(bnlp::InplaceBatchNLPModel, xs) =
  _batch_map(jac, bnlp, xs)
batch_jac_lin(bnlp::InplaceBatchNLPModel, xs) =
  _batch_map(jac_lin, bnlp, xs)
batch_jac_nln(bnlp::InplaceBatchNLPModel, xs) =
  _batch_map(jac_nln, bnlp, xs)
batch_jac_lin_coord(bnlp::InplaceBatchNLPModel, xs) =
  _batch_map(jac_lin_coord, bnlp, xs)
batch_jac_coord(bnlp::InplaceBatchNLPModel, xs) =
  _batch_map(jac_coord, bnlp, xs)
batch_jac_nln_coord(bnlp::InplaceBatchNLPModel, xs) =
  _batch_map(jac_nln_coord, bnlp, xs)
batch_varscale(bnlp::InplaceBatchNLPModel) =
  _batch_map(varscale, bnlp)
batch_lagscale(bnlp::InplaceBatchNLPModel) =
  _batch_map(lagscale, bnlp)
batch_conscale(bnlp::InplaceBatchNLPModel) =
  _batch_map(conscale, bnlp)
batch_jprod(bnlp::InplaceBatchNLPModel, xs, vs) =
  _batch_map(jprod, bnlp, xs, vs)
batch_jtprod(bnlp::InplaceBatchNLPModel, xs, vs) =
  _batch_map(jtprod, bnlp, xs, vs)
batch_jprod_nln(bnlp::InplaceBatchNLPModel, xs, vs) =
  _batch_map(jprod_nln, bnlp, xs, vs)
batch_jtprod_nln(bnlp::InplaceBatchNLPModel, xs, vs) =
  _batch_map(jtprod_nln, bnlp, xs, vs)
batch_jprod_lin(bnlp::InplaceBatchNLPModel, xs, vs) =
  _batch_map(jprod_lin, bnlp, xs, vs)
batch_jtprod_lin(bnlp::InplaceBatchNLPModel, xs, vs) =
  _batch_map(jtprod_lin, bnlp, xs, vs)
batch_ghjvprod(bnlp::InplaceBatchNLPModel, xs, gs, vs) =
  _batch_map(ghjvprod, bnlp, xs, gs, vs)

batch_jac_structure!(bnlp::InplaceBatchNLPModel, rowss, colss) =
  _batch_map!(jac_structure!, bnlp, rowss, colss)
batch_jac_lin_structure!(bnlp::InplaceBatchNLPModel, rowss, colss) =
  _batch_map!(jac_lin_structure!, bnlp, rowss, colss)
batch_jac_nln_structure!(bnlp::InplaceBatchNLPModel, rowss, colss) =
  _batch_map!(jac_nln_structure!, bnlp, rowss, colss)
batch_hess_structure!(bnlp::InplaceBatchNLPModel, rowss, colss) =
  _batch_map!(hess_structure!, bnlp, rowss, colss)
batch_jac_lin_coord!(bnlp::InplaceBatchNLPModel, xs, valss) =
  _batch_map!(jac_lin_coord!, bnlp, xs, valss)
batch_grad!(bnlp::InplaceBatchNLPModel, xs, gs) =
  _batch_map!(grad!, bnlp, xs, gs)
batch_cons!(bnlp::InplaceBatchNLPModel, xs, cs) =
  _batch_map!(cons!, bnlp, xs, cs)
batch_cons_lin!(bnlp::InplaceBatchNLPModel, xs, cs) =
  _batch_map!(cons_lin!, bnlp, xs, cs)
batch_cons_nln!(bnlp::InplaceBatchNLPModel, xs, cs) =
  _batch_map!(cons_nln!, bnlp, xs, cs)
batch_jac_coord!(bnlp::InplaceBatchNLPModel, xs, valss) =
  _batch_map!(jac_coord!, bnlp, xs, valss)
batch_jac_nln_coord!(bnlp::InplaceBatchNLPModel, xs, valss) =
  _batch_map!(jac_nln_coord!, bnlp, xs, valss)
batch_jprod!(bnlp::InplaceBatchNLPModel, xs, vs, Jvs) =
  _batch_map!(jprod!, bnlp, xs, vs, Jvs)
batch_jtprod!(bnlp::InplaceBatchNLPModel, xs, vs, Jtvs) =
  _batch_map!(jtprod!, bnlp, xs, vs, Jtvs)
batch_jprod_nln!(bnlp::InplaceBatchNLPModel, xs, vs, Jvs) =
  _batch_map!(jprod_nln!, bnlp, xs, vs, Jvs)
batch_jtprod_nln!(bnlp::InplaceBatchNLPModel, xs, vs, Jtvs) =
  _batch_map!(jtprod_nln!, bnlp, xs, vs, Jtvs)
batch_jprod_lin!(bnlp::InplaceBatchNLPModel, xs, vs, Jvs) =
  _batch_map!(jprod_lin!, bnlp, xs, vs, Jvs)
batch_jtprod_lin!(bnlp::InplaceBatchNLPModel, xs, vs, Jtvs) =
  _batch_map!(jtprod_lin!, bnlp, xs, vs, Jtvs)
batch_ghjvprod!(bnlp::InplaceBatchNLPModel, xs, gs, vs, gHvs) =
  _batch_map!(ghjvprod!, bnlp, xs, gs, vs, gHvs)

## jth  FIXME: allow for vector of js
batch_jth_con(bnlp::InplaceBatchNLPModel, xs, j::Integer) =
  _batch_map((m, x) -> jth_con(m, x, j), bnlp, xs)
batch_jth_congrad(bnlp::InplaceBatchNLPModel, xs, j::Integer) =
  _batch_map((m, x) -> jth_congrad(m, x, j), bnlp, xs)
batch_jth_sparse_congrad(bnlp::InplaceBatchNLPModel, xs, j::Integer) =
  _batch_map((m, x) -> jth_sparse_congrad(m, x, j), bnlp, xs)
batch_jth_hess_coord(bnlp::InplaceBatchNLPModel, xs, j::Integer) =
  _batch_map((m, x) -> jth_hess_coord(m, x, j), bnlp, xs)
batch_jth_hess(bnlp::InplaceBatchNLPModel, xs, j::Integer) =
  _batch_map((m, x) -> jth_hess(m, x, j), bnlp, xs)
batch_jth_hprod(bnlp::InplaceBatchNLPModel, xs, vs, j::Integer) =
  _batch_map((m, x, v) -> jth_hprod(m, x, v, j), bnlp, xs, vs)

batch_jth_congrad!(bnlp::InplaceBatchNLPModel, xs, j::Integer, outputs) =
  _batch_map!((m, x, out) -> jth_congrad!(m, x, j, out), bnlp, xs, outputs)
batch_jth_hess_coord!(bnlp::InplaceBatchNLPModel, xs, j::Integer, outputs) =
  _batch_map!((m, x, out) -> jth_hess_coord!(m, x, j, out), bnlp, xs, outputs)
batch_jth_hprod!(bnlp::InplaceBatchNLPModel, xs, vs, j::Integer, outputs) =
  _batch_map!((m, x, v, out) -> jth_hprod!(m, x, v, j, out), bnlp, xs, vs, outputs)

# hess (need to treat obj_weight)  FIXME: obj_weights is required in batch API
batch_hprod(bnlp::InplaceBatchNLPModel, xs, vs; obj_weights) =
  _batch_map_weight(hprod, bnlp, obj_weights, xs, vs)
batch_hprod(bnlp::InplaceBatchNLPModel, xs, ys, vs; obj_weights) =
  _batch_map_weight(hprod, bnlp, obj_weights, xs, ys, vs)
batch_hess_coord(bnlp::InplaceBatchNLPModel, xs; obj_weights) =
  _batch_map_weight(hess_coord, bnlp, obj_weights, xs)
batch_hess_coord(bnlp::InplaceBatchNLPModel, xs, ys; obj_weights) =
  _batch_map_weight(hess_coord, bnlp, obj_weights, xs, ys)
batch_hess_op(bnlp::InplaceBatchNLPModel, xs; obj_weights) = _inplace_operator_error()
batch_hess_op(bnlp::InplaceBatchNLPModel, xs, ys; obj_weights) = _inplace_operator_error()

batch_hprod!(bnlp::InplaceBatchNLPModel, xs, vs, outputs; obj_weights) =
  _batch_map_weight!(hprod!, bnlp, obj_weights, xs, vs, outputs)
batch_hprod!(bnlp::InplaceBatchNLPModel, xs, ys, vs, outputs; obj_weights) =
  _batch_map_weight!(hprod!, bnlp, obj_weights, xs, ys, vs, outputs)
batch_hess_coord!(bnlp::InplaceBatchNLPModel, xs, outputs; obj_weights) =
  _batch_map_weight!(hess_coord!, bnlp, obj_weights, xs, outputs)
batch_hess_coord!(bnlp::InplaceBatchNLPModel, xs, ys, outputs; obj_weights) =
  _batch_map_weight!(hess_coord!, bnlp, obj_weights, xs, ys, outputs)
batch_hess_op!(bnlp::InplaceBatchNLPModel, xs, Hvs; obj_weights) = _inplace_operator_error()
batch_hess_op!(bnlp::InplaceBatchNLPModel, xs, ys, Hvs; obj_weights) = _inplace_operator_error()

batch_hess(bnlp::InplaceBatchNLPModel, xs; obj_weights) =
  _batch_map_weight(hess, bnlp, obj_weights, xs)
batch_hess(bnlp::InplaceBatchNLPModel, xs, ys; obj_weights) =
  _batch_map_weight(hess, bnlp, obj_weights, xs, ys)

## operators
batch_jac_op(bnlp::InplaceBatchNLPModel, xs) = _inplace_operator_error()
batch_jac_lin_op(bnlp::InplaceBatchNLPModel, xs) = _inplace_operator_error()
batch_jac_nln_op(bnlp::InplaceBatchNLPModel, xs) = _inplace_operator_error()

batch_jac_op!(bnlp::InplaceBatchNLPModel, xs, Jvs, Jtvs) = _inplace_operator_error()
batch_jac_lin_op!(bnlp::InplaceBatchNLPModel, xs, Jvs, Jtvs) = _inplace_operator_error()
batch_jac_nln_op!(bnlp::InplaceBatchNLPModel, xs, Jvs, Jtvs) = _inplace_operator_error()

## tuple functions
batch_objgrad(bnlp::InplaceBatchNLPModel, xs) =
  _batch_map_tuple(objgrad, bnlp, xs)
batch_objcons(bnlp::InplaceBatchNLPModel, xs) =
  _batch_map_tuple(objcons, bnlp, xs)

batch_objgrad!(bnlp::InplaceBatchNLPModel, xs, gs) =
  _batch_map_tuple!(objgrad!, bnlp, gs, xs)
batch_objcons!(bnlp::InplaceBatchNLPModel, xs, cs) =
  _batch_map_tuple!(objcons!, bnlp, cs, xs)
