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

function _batch_map!(f::F, bnlp::InplaceBatchNLPModel, outputs, xs::Vararg{T,N}) where {F,T,N}
  n = bnlp.batch_size
  @lencheck_tup n xs
  @lencheck n outputs
  for i = 1:n
    args_i = (x[i] for x in xs)
    bnlp.updates[i](bnlp.base_model)  # call update function
    f(bnlp.base_model, outputs[i], args_i...)
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

function _batch_map_weight!(f::F, bnlp::InplaceBatchNLPModel, outputs, obj_weights, xs::Vararg{T,N}) where {F,T,N}
  n = bnlp.batch_size
  @lencheck_tup n xs
  @lencheck n outputs obj_weights
  for i = 1:n
    args_i = (x[i] for x in xs)
    bnlp.updates[i](bnlp.base_model)  # call update function
    f(bnlp.base_model, outputs[i], args_i...; obj_weight = obj_weights[i])
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
batch_jac_lin(bnlp::InplaceBatchNLPModel) =
  _batch_map(jac_lin, bnlp)
batch_jac_nln(bnlp::InplaceBatchNLPModel, xs) =
  _batch_map(jac_nln, bnlp, xs)
batch_jac_lin_coord(bnlp::InplaceBatchNLPModel) =
  _batch_map(jac_lin_coord, bnlp)
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
batch_jprod_lin(bnlp::InplaceBatchNLPModel, vs) =
  _batch_map(jprod_lin, bnlp, vs)
batch_jtprod_lin(bnlp::InplaceBatchNLPModel, vs) =
  _batch_map(jtprod_lin, bnlp, vs)
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
batch_jac_lin_coord!(bnlp::InplaceBatchNLPModel, valss) =
  _batch_map!(jac_lin_coord!, bnlp, valss)
batch_grad!(bnlp::InplaceBatchNLPModel, xs, gs) =
  _batch_map!((m, g, x) -> grad!(m, x, g), bnlp, gs, xs)
batch_cons!(bnlp::InplaceBatchNLPModel, xs, cs) =
  _batch_map!((m, c, x) -> cons!(m, x, c), bnlp, cs, xs)
batch_cons_lin!(bnlp::InplaceBatchNLPModel, xs, cs) =
  _batch_map!((m, c, x) -> cons_lin!(m, x, c), bnlp, cs, xs)
batch_cons_nln!(bnlp::InplaceBatchNLPModel, xs, cs) =
  _batch_map!((m, c, x) -> cons_nln!(m, x, c), bnlp, cs, xs)
batch_jac_coord!(bnlp::InplaceBatchNLPModel, xs, valss) =
  _batch_map!((m, vals, x) -> jac_coord!(m, x, vals), bnlp, valss, xs)
batch_jac_nln_coord!(bnlp::InplaceBatchNLPModel, xs, valss) =
  _batch_map!((m, vals, x) -> jac_nln_coord!(m, x, vals), bnlp, valss, xs)
batch_jprod!(bnlp::InplaceBatchNLPModel, xs, vs, Jvs) =
  _batch_map!((m, Jv, x, v) -> jprod!(m, x, v, Jv), bnlp, Jvs, xs, vs)
batch_jtprod!(bnlp::InplaceBatchNLPModel, xs, vs, Jtvs) =
  _batch_map!((m, Jtv, x, v) -> jtprod!(m, x, v, Jtv), bnlp, Jtvs, xs, vs)
batch_jprod_nln!(bnlp::InplaceBatchNLPModel, xs, vs, Jvs) =
  _batch_map!((m, Jv, x, v) -> jprod_nln!(m, x, v, Jv), bnlp, Jvs, xs, vs)
batch_jtprod_nln!(bnlp::InplaceBatchNLPModel, xs, vs, Jtvs) =
  _batch_map!((m, Jtv, x, v) -> jtprod_nln!(m, x, v, Jtv), bnlp, Jtvs, xs, vs)
batch_jprod_lin!(bnlp::InplaceBatchNLPModel, vs, Jvs) =
  _batch_map!((m, Jv, v) -> jprod_lin!(m, v, Jv), bnlp, Jvs, vs)
batch_jtprod_lin!(bnlp::InplaceBatchNLPModel, vs, Jtvs) =
  _batch_map!((m, Jtv, v) -> jtprod_lin!(m, v, Jtv), bnlp, Jtvs, vs)
batch_ghjvprod!(bnlp::InplaceBatchNLPModel, xs, gs, vs, gHvs) =
  _batch_map!((m, gHv, x, g, v) -> ghjvprod!(m, x, g, v, gHv), bnlp, gHvs, xs, gs, vs)

## jth
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
  _batch_map!((m, out, x) -> jth_congrad!(m, x, j, out), bnlp, outputs, xs)
batch_jth_hess_coord!(bnlp::InplaceBatchNLPModel, xs, j::Integer, outputs) =
  _batch_map!((m, out, x) -> jth_hess_coord!(m, x, j, out), bnlp, outputs, xs)
batch_jth_hprod!(bnlp::InplaceBatchNLPModel, xs, vs, j::Integer, outputs) =
  _batch_map!((m, out, x, v) -> jth_hprod!(m, x, v, j, out), bnlp, outputs, xs, vs)

# hess (need to treat obj_weight)  FIXME: obj_weights is required in batch API
batch_hprod(bnlp::InplaceBatchNLPModel, xs, vs; obj_weights) =
  _batch_map_weight((m, x, v; obj_weight) -> hprod(m, x, v; obj_weight = obj_weight), bnlp, obj_weights, xs, vs)
batch_hprod(bnlp::InplaceBatchNLPModel, xs, ys, vs; obj_weights) =
  _batch_map_weight((m, x, y, v; obj_weight) -> hprod(m, x, y, v; obj_weight = obj_weight), bnlp, obj_weights, xs, ys, vs)
batch_hess_coord(bnlp::InplaceBatchNLPModel, xs; obj_weights) =
  _batch_map_weight((m, x; obj_weight) -> hess_coord(m, x; obj_weight = obj_weight), bnlp, obj_weights, xs)
batch_hess_coord(bnlp::InplaceBatchNLPModel, xs, ys; obj_weights) =
  _batch_map_weight((m, x, y; obj_weight) -> hess_coord(m, x, y; obj_weight = obj_weight), bnlp, obj_weights, xs, ys)
batch_hess_op(bnlp::InplaceBatchNLPModel, xs; obj_weights) =
  _batch_map_weight((m, x; obj_weight) -> hess_op(m, x; obj_weight = obj_weight), bnlp, obj_weights, xs)
batch_hess_op(bnlp::InplaceBatchNLPModel, xs, ys; obj_weights) =
  _batch_map_weight((m, x, y; obj_weight) -> hess_op(m, x, y; obj_weight = obj_weight), bnlp, obj_weights, xs, ys)

batch_hprod!(bnlp::InplaceBatchNLPModel, xs, vs, outputs; obj_weights) =
  _batch_map_weight!((m, Hv, x, v; obj_weight) -> hprod!(m, x, v, Hv; obj_weight = obj_weight), bnlp, outputs, obj_weights, xs, vs)
batch_hprod!(bnlp::InplaceBatchNLPModel, xs, ys, vs, outputs; obj_weights) =
  _batch_map_weight!((m, Hv, x, y, v; obj_weight) -> hprod!(m, x, y, v, Hv; obj_weight = obj_weight), bnlp, outputs, obj_weights, xs, ys, vs)
batch_hess_coord!(bnlp::InplaceBatchNLPModel, xs, outputs; obj_weights) =
  _batch_map_weight!((m, vals, x; obj_weight) -> hess_coord!(m, x, vals; obj_weight = obj_weight), bnlp, outputs, obj_weights, xs)
batch_hess_coord!(bnlp::InplaceBatchNLPModel, xs, ys, outputs; obj_weights) =
  _batch_map_weight!((m, vals, x, y; obj_weight) -> hess_coord!(m, x, y, vals; obj_weight = obj_weight), bnlp, outputs, obj_weights, xs, ys)
batch_hess_op!(bnlp::InplaceBatchNLPModel, xs, Hvs; obj_weights) =
  _batch_map_weight((m, x, Hv; obj_weight) -> hess_op!(m, x, Hv; obj_weight = obj_weight), bnlp, obj_weights, xs, Hvs)
batch_hess_op!(bnlp::InplaceBatchNLPModel, xs, ys, Hvs; obj_weights) =
  _batch_map_weight((m, x, y, Hv; obj_weight) -> hess_op!(m, x, y, Hv; obj_weight = obj_weight), bnlp, obj_weights, xs, ys, Hvs)

batch_hess(bnlp::InplaceBatchNLPModel, xs; obj_weights) =
  _batch_map_weight((m, x; obj_weight) -> hess(m, x; obj_weight = obj_weight), bnlp, obj_weights, xs)
batch_hess(bnlp::InplaceBatchNLPModel, xs, ys; obj_weights) =
  _batch_map_weight((m, x, y; obj_weight) -> hess(m, x, y; obj_weight = obj_weight), bnlp, obj_weights, xs, ys)

## operators
batch_jac_op(bnlp::InplaceBatchNLPModel, xs) =
  _batch_map(jac_op, bnlp, xs)
batch_jac_lin_op(bnlp::InplaceBatchNLPModel) =
  _batch_map(jac_lin_op, bnlp)
batch_jac_nln_op(bnlp::InplaceBatchNLPModel, xs) =
  _batch_map(jac_nln_op, bnlp, xs)

batch_jac_op!(bnlp::InplaceBatchNLPModel, xs, Jvs, Jtvs) =
  _batch_map(jac_op!, bnlp, xs, Jvs, Jtvs)
batch_jac_lin_op!(bnlp::InplaceBatchNLPModel, Jvs, Jtvs) =
  _batch_map(jac_lin_op!, bnlp, Jvs, Jtvs)
batch_jac_nln_op!(bnlp::InplaceBatchNLPModel, xs, Jvs, Jtvs) =
  _batch_map(jac_nln_op!, bnlp, xs, Jvs, Jtvs)

## tuple functions
batch_objgrad(bnlp::InplaceBatchNLPModel, xs) =
  _batch_map_tuple(objgrad, bnlp, xs)
batch_objcons(bnlp::InplaceBatchNLPModel, xs) =
  _batch_map_tuple(objcons, bnlp, xs)

batch_objgrad!(bnlp::InplaceBatchNLPModel, xs, gs) =
  _batch_map_tuple!(objgrad!, bnlp, gs, xs)
batch_objcons!(bnlp::InplaceBatchNLPModel, xs, cs) =
  _batch_map_tuple!(objcons!, bnlp, cs, xs)
