const VV = Vector{<:AbstractVector}

export AbstractBatchNLPModel
export batch_obj, batch_grad, batch_grad!, batch_objgrad, batch_objgrad!, batch_objcons, batch_objcons!
export batch_cons, batch_cons!, batch_cons_lin, batch_cons_lin!, batch_cons_nln, batch_cons_nln!
export batch_jth_con, batch_jth_congrad, batch_jth_congrad!, batch_jth_sparse_congrad
export batch_jac_structure!, batch_jac_structure, batch_jac_coord!, batch_jac_coord
export batch_jac, batch_jprod, batch_jprod!, batch_jtprod, batch_jtprod!, batch_jac_op, batch_jac_op!
export batch_jac_lin_structure!, batch_jac_lin_structure, batch_jac_lin_coord!, batch_jac_lin_coord
export batch_jac_lin, batch_jprod_lin, batch_jprod_lin!, batch_jtprod_lin, batch_jtprod_lin!, batch_jac_lin_op, batch_jac_lin_op!
export batch_jac_nln_structure!, batch_jac_nln_structure, batch_jac_nln_coord!, batch_jac_nln_coord
export batch_jac_nln, batch_jprod_nln, batch_jprod_nln!, batch_jtprod_nln, batch_jtprod_nln!, batch_jac_nln_op, batch_jac_nln_op!
export batch_jth_hess_coord, batch_jth_hess_coord!, batch_jth_hess
export batch_jth_hprod, batch_jth_hprod!, batch_ghjvprod, batch_ghjvprod!
export batch_hess_structure!, batch_hess_structure, batch_hess_coord!, batch_hess_coord
export batch_hess, batch_hprod, batch_hprod!, batch_hess_op, batch_hess_op!
export batch_varscale, batch_lagscale, batch_conscale

abstract type AbstractBatchNLPModel{T, S} end

## base api
batch_jac_structure(bnlp::AbstractBatchNLPModel) =
  jac_structure(first(bnlp))
batch_jac_lin_structure(bnlp::AbstractBatchNLPModel) =
  jac_lin_structure(first(bnlp))
batch_jac_nln_structure(bnlp::AbstractBatchNLPModel) =
  jac_nln_structure(first(bnlp))
batch_hess_structure(bnlp::AbstractBatchNLPModel) =
  hess_structure(first(bnlp))
batch_jac_structure!(bnlp::AbstractBatchNLPModel, rows, cols) =
  jac_structure!(first(bnlp), rows, cols)
batch_jac_lin_structure!(bnlp::AbstractBatchNLPModel, rows, cols) =
  jac_lin_structure!(first(bnlp), rows, cols)
batch_jac_nln_structure!(bnlp::AbstractBatchNLPModel, rows, cols) =
  jac_nln_structure!(first(bnlp), rows, cols)
batch_hess_structure!(bnlp::AbstractBatchNLPModel, rows, cols) =
  hess_structure!(first(bnlp), rows, cols)
batch_obj(bnlp::AbstractBatchNLPModel, xs::VV) =
  _batch_map(obj, bnlp, xs)
batch_grad(bnlp::AbstractBatchNLPModel, xs::VV) =
  _batch_map(grad, bnlp, xs)
batch_cons(bnlp::AbstractBatchNLPModel, xs::VV) =
  _batch_map(cons, bnlp, xs)
batch_cons_lin(bnlp::AbstractBatchNLPModel, xs::VV) =
  _batch_map(cons_lin, bnlp, xs)
batch_cons_nln(bnlp::AbstractBatchNLPModel, xs::VV) =
  _batch_map(cons_nln, bnlp, xs)
batch_jac(bnlp::AbstractBatchNLPModel, xs::VV) =
  _batch_map(jac, bnlp, xs)
batch_jac_lin(bnlp::AbstractBatchNLPModel) =
  _batch_map(jac_lin, bnlp)
batch_jac_nln(bnlp::AbstractBatchNLPModel, xs::VV) =
  _batch_map(jac_nln, bnlp, xs)
batch_jac_lin_coord(bnlp::AbstractBatchNLPModel) =
  _batch_map(jac_lin_coord, bnlp)
batch_jac_coord(bnlp::AbstractBatchNLPModel, xs::VV) =
  _batch_map(jac_coord, bnlp, xs)
batch_jac_nln_coord(bnlp::AbstractBatchNLPModel, xs::VV) =
  _batch_map(jac_nln_coord, bnlp, xs)
batch_varscale(bnlp::AbstractBatchNLPModel) =
  _batch_map(varscale, bnlp)
batch_lagscale(bnlp::AbstractBatchNLPModel) =
  _batch_map(lagscale, bnlp)
batch_conscale(bnlp::AbstractBatchNLPModel) =
  _batch_map(conscale, bnlp)
batch_jprod(bnlp::AbstractBatchNLPModel, xs::VV, vs::VV) =
  _batch_map(jprod, bnlp, xs, vs)
batch_jtprod(bnlp::AbstractBatchNLPModel, xs::VV, vs::VV) =
  _batch_map(jtprod, bnlp, xs, vs)
batch_jprod_nln(bnlp::AbstractBatchNLPModel, xs::VV, vs::VV) =
  _batch_map(jprod_nln, bnlp, xs, vs)
batch_jtprod_nln(bnlp::AbstractBatchNLPModel, xs::VV, vs::VV) =
  _batch_map(jtprod_nln, bnlp, xs, vs)
batch_jprod_lin(bnlp::AbstractBatchNLPModel, vs::VV) =
  _batch_map(jprod_lin, bnlp, vs)
batch_jtprod_lin(bnlp::AbstractBatchNLPModel, vs::VV) =
  _batch_map(jtprod_lin, bnlp, vs)
batch_ghjvprod(bnlp::AbstractBatchNLPModel, xs::VV, gs::VV, vs::VV) =
  _batch_map(ghjvprod, bnlp, xs, gs, vs)

batch_grad!(bnlp::AbstractBatchNLPModel, xs::VV, gs::Vector) =
  _batch_map!((m, g, x) -> grad!(m, x, g), bnlp, gs, xs)
batch_cons!(bnlp::AbstractBatchNLPModel, xs::VV, cs::Vector) =
  _batch_map!((m, c, x) -> cons!(m, x, c), bnlp, cs, xs)
batch_cons_lin!(bnlp::AbstractBatchNLPModel, xs::VV, cs::Vector) =
  _batch_map!((m, c, x) -> cons_lin!(m, x, c), bnlp, cs, xs)
batch_cons_nln!(bnlp::AbstractBatchNLPModel, xs::VV, cs::Vector) =
  _batch_map!((m, c, x) -> cons_nln!(m, x, c), bnlp, cs, xs)
batch_jac_lin_coord!(bnlp::AbstractBatchNLPModel, valss::Vector) =
  _batch_map!((m, vals) -> jac_lin_coord!(m, vals), bnlp, valss)
batch_jac_coord!(bnlp::AbstractBatchNLPModel, xs::VV, valss::Vector) =
  _batch_map!((m, vals, x) -> jac_coord!(m, x, vals), bnlp, valss, xs)
batch_jac_nln_coord!(bnlp::AbstractBatchNLPModel, xs::VV, valss::Vector) =
  _batch_map!((m, vals, x) -> jac_nln_coord!(m, x, vals), bnlp, valss, xs)
batch_jprod!(bnlp::AbstractBatchNLPModel, xs::VV, vs::VV, Jvs::Vector) =
  _batch_map!((m, Jv, x, v) -> jprod!(m, x, v, Jv), bnlp, Jvs, xs, vs)
batch_jtprod!(bnlp::AbstractBatchNLPModel, xs::VV, vs::VV, Jtvs::Vector) =
  _batch_map!((m, Jtv, x, v) -> jtprod!(m, x, v, Jtv), bnlp, Jtvs, xs, vs)
batch_jprod_nln!(bnlp::AbstractBatchNLPModel, xs::VV, vs::VV, Jvs::Vector) =
  _batch_map!((m, Jv, x, v) -> jprod_nln!(m, x, v, Jv), bnlp, Jvs, xs, vs)
batch_jtprod_nln!(bnlp::AbstractBatchNLPModel, xs::VV, vs::VV, Jtvs::Vector) =
  _batch_map!((m, Jtv, x, v) -> jtprod_nln!(m, x, v, Jtv), bnlp, Jtvs, xs, vs)
batch_jprod_lin!(bnlp::AbstractBatchNLPModel, vs::VV, Jvs::Vector) =
  _batch_map!((m, Jv, v) -> jprod_lin!(m, v, Jv), bnlp, Jvs, vs)
batch_jtprod_lin!(bnlp::AbstractBatchNLPModel, vs::VV, Jtvs::Vector) =
  _batch_map!((m, Jtv, v) -> jtprod_lin!(m, v, Jtv), bnlp, Jtvs, vs)
batch_ghjvprod!(bnlp::AbstractBatchNLPModel, xs::VV, gs::VV, vs::VV, gHvs::Vector) =
  _batch_map!((m, gHv, x, g, v) -> ghjvprod!(m, x, g, v, gHv), bnlp, gHvs, xs, gs, vs)

## jth
batch_jth_con(bnlp::AbstractBatchNLPModel, xs::VV, j::Integer) =
  _batch_map((m, x) -> jth_con(m, x, j), bnlp, xs)
batch_jth_congrad(bnlp::AbstractBatchNLPModel, xs::VV, j::Integer) =
  _batch_map((m, x) -> jth_congrad(m, x, j), bnlp, xs)
batch_jth_sparse_congrad(bnlp::AbstractBatchNLPModel, xs::VV, j::Integer) =
  _batch_map((m, x) -> jth_sparse_congrad(m, x, j), bnlp, xs)
batch_jth_hess_coord(bnlp::AbstractBatchNLPModel, xs::VV, j::Integer) =
  _batch_map((m, x) -> jth_hess_coord(m, x, j), bnlp, xs)
batch_jth_hess(bnlp::AbstractBatchNLPModel, xs::VV, j::Integer) =
  _batch_map((m, x) -> jth_hess(m, x, j), bnlp, xs)
batch_jth_hprod(bnlp::AbstractBatchNLPModel, xs::VV, vs::VV, j::Integer) =
  _batch_map((m, x, v) -> jth_hprod(m, x, v, j), bnlp, xs, vs)

batch_jth_congrad!(bnlp::AbstractBatchNLPModel, xs::VV, j::Integer, outputs::Vector) =
  _batch_map!((m, out, x) -> jth_congrad!(m, x, j, out), bnlp, outputs, xs)
batch_jth_hess_coord!(bnlp::AbstractBatchNLPModel, xs::VV, j::Integer, outputs::Vector) =
  _batch_map!((m, out, x) -> jth_hess_coord!(m, x, j, out), bnlp, outputs, xs)
batch_jth_hprod!(bnlp::AbstractBatchNLPModel, xs::VV, vs::VV, j::Integer, outputs::Vector) =
  _batch_map!((m, out, x, v) -> jth_hprod!(m, x, v, j, out), bnlp, outputs, xs, vs)

# hess (need to treat obj_weight)
batch_hprod(bnlp::AbstractBatchNLPModel{T, S}, xs::VV, vs::VV; obj_weights::Vector{<:Real} = ones(T, length(bnlp))) where {T, S} =
  _batch_map_weight((m, x, v; obj_weight) -> hprod(m, x, v; obj_weight = obj_weight), bnlp, obj_weights, xs, vs)
batch_hprod(bnlp::AbstractBatchNLPModel{T, S}, xs::VV, ys::VV, vs::VV; obj_weights::Vector{<:Real} = ones(T, length(bnlp))) where {T, S} =
  _batch_map_weight((m, x, y, v; obj_weight) -> hprod(m, x, y, v; obj_weight = obj_weight), bnlp, obj_weights, xs, ys, vs)
batch_hess_coord(bnlp::AbstractBatchNLPModel{T, S}, xs::VV; obj_weights::Vector{<:Real} = ones(T, length(bnlp))) where {T, S} =
  _batch_map_weight((m, x; obj_weight) -> hess_coord(m, x; obj_weight = obj_weight), bnlp, obj_weights, xs)
batch_hess_coord(bnlp::AbstractBatchNLPModel{T, S}, xs::VV, ys::VV; obj_weights::Vector{<:Real} = ones(T, length(bnlp))) where {T, S} =
  _batch_map_weight((m, x, y; obj_weight) -> hess_coord(m, x, y; obj_weight = obj_weight), bnlp, obj_weights, xs, ys)
batch_hess_op(bnlp::AbstractBatchNLPModel{T, S}, xs::VV; obj_weights::Vector{<:Real} = ones(T, length(bnlp))) where {T, S} =
  _batch_map_weight((m, x; obj_weight) -> hess_op(m, x; obj_weight = obj_weight), bnlp, obj_weights, xs)
batch_hess_op(bnlp::AbstractBatchNLPModel{T, S}, xs::VV, ys::VV; obj_weights::Vector{<:Real} = ones(T, length(bnlp))) where {T, S} =
  _batch_map_weight((m, x, y; obj_weight) -> hess_op(m, x, y; obj_weight = obj_weight), bnlp, obj_weights, xs, ys)

batch_hprod!(bnlp::AbstractBatchNLPModel{T, S}, xs::VV, vs::VV, outputs::Vector; obj_weights::Vector{<:Real} = ones(T, length(bnlp))) where {T, S} =
  _batch_map_weight!((m, Hv, x, v; obj_weight) -> hprod!(m, x, v, Hv; obj_weight = obj_weight), bnlp, outputs, obj_weights, xs, vs)
batch_hprod!(bnlp::AbstractBatchNLPModel{T, S}, xs::VV, ys::VV, vs::VV, outputs::Vector; obj_weights::Vector{<:Real} = ones(T, length(bnlp))) where {T, S} =
  _batch_map_weight!((m, Hv, x, y, v; obj_weight) -> hprod!(m, x, y, v, Hv; obj_weight = obj_weight), bnlp, outputs, obj_weights, xs, ys, vs)
batch_hess_coord!(bnlp::AbstractBatchNLPModel{T, S}, xs::VV, outputs::Vector; obj_weights::Vector{<:Real} = ones(T, length(bnlp))) where {T, S} =
  _batch_map_weight!((m, vals, x; obj_weight) -> hess_coord!(m, x, vals; obj_weight = obj_weight), bnlp, outputs, obj_weights, xs)
batch_hess_coord!(bnlp::AbstractBatchNLPModel{T, S}, xs::VV, ys::VV, outputs::Vector; obj_weights::Vector{<:Real} = ones(T, length(bnlp))) where {T, S} =
  _batch_map_weight!((m, vals, x, y; obj_weight) -> hess_coord!(m, x, y, vals; obj_weight = obj_weight), bnlp, outputs, obj_weights, xs, ys)
batch_hess_op!(bnlp::AbstractBatchNLPModel{T, S}, xs::VV, Hvs::Vector; obj_weights::Vector{<:Real} = ones(T, length(bnlp))) where {T, S} =
  _batch_map_weight((m, x, Hv; obj_weight) -> hess_op!(m, x, Hv; obj_weight = obj_weight), bnlp, obj_weights, xs, Hvs)
batch_hess_op!(bnlp::AbstractBatchNLPModel{T, S}, xs::VV, ys::VV, Hvs::Vector; obj_weights::Vector{<:Real} = ones(T, length(bnlp))) where {T, S} =
  _batch_map_weight((m, x, y, Hv; obj_weight) -> hess_op!(m, x, y, Hv; obj_weight = obj_weight), bnlp, obj_weights, xs, ys, Hvs)

batch_hess(bnlp::AbstractBatchNLPModel{T, S}, xs::VV; obj_weights::Vector{<:Real} = ones(T, length(bnlp))) where {T, S} =
  _batch_map_weight((m, x; obj_weight) -> hess(m, x; obj_weight = obj_weight), bnlp, obj_weights, xs)
batch_hess(bnlp::AbstractBatchNLPModel{T, S}, xs::VV, ys::VV; obj_weights::Vector{<:Real} = ones(T, length(bnlp))) where {T, S} =
  _batch_map_weight((m, x, y; obj_weight) -> hess(m, x, y; obj_weight = obj_weight), bnlp, obj_weights, xs, ys)

## operators
batch_jac_op(bnlp::AbstractBatchNLPModel, xs::VV) =
  _batch_map(jac_op, bnlp, xs)
batch_jac_lin_op(bnlp::AbstractBatchNLPModel) =
  _batch_map(jac_lin_op, bnlp)
batch_jac_nln_op(bnlp::AbstractBatchNLPModel, xs::VV) =
  _batch_map(jac_nln_op, bnlp, xs)

batch_jac_op!(bnlp::AbstractBatchNLPModel, xs::VV, Jvs::Vector, Jtvs::Vector) =
  _batch_map((m, x, Jv, Jtv) -> jac_op!(m, x, Jv, Jtv), bnlp, xs, Jvs, Jtvs)
batch_jac_lin_op!(bnlp::AbstractBatchNLPModel, Jvs::Vector, Jtvs::Vector) =
  _batch_map((m, Jv, Jtv) -> jac_lin_op!(m, Jv, Jtv), bnlp, Jvs, Jtvs)
batch_jac_nln_op!(bnlp::AbstractBatchNLPModel, xs::VV, Jvs::Vector, Jtvs::Vector) =
  _batch_map((m, x, Jv, Jtv) -> jac_nln_op!(m, x, Jv, Jtv), bnlp, xs, Jvs, Jtvs)

## tuple functions
batch_objgrad(bnlp::AbstractBatchNLPModel, xs::VV) =
  _batch_map_tuple(objgrad, bnlp, xs)
batch_objcons(bnlp::AbstractBatchNLPModel, xs::VV) =
  _batch_map_tuple(objcons, bnlp, xs)

batch_objgrad!(bnlp::AbstractBatchNLPModel, xs::VV, gs::Vector) =
  _batch_map_tuple!(objgrad!, bnlp, gs, xs)
batch_objcons!(bnlp::AbstractBatchNLPModel, xs::VV, cs::Vector) =
  _batch_map_tuple!(objcons!, bnlp, cs, xs)

function NLPModels.increment!(bnlp::AbstractBatchNLPModel, fun::Symbol)
  NLPModels.increment!(bnlp, Val(fun))
end
  