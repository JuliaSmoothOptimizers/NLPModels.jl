export VectorBatchNLPModel
struct VectorBatchNLPModel{T, S, M <: AbstractNLPModel{T, S}} <: AbstractBatchNLPModel{T, S}
  models::Vector{M}
  counters::Counters
  meta::NLPModelMeta{T, S}
end
function VectorBatchNLPModel(models::Vector{M}) where {M <: AbstractNLPModel}
  isempty(models) && error("Cannot create VectorBatchNLPModel from empty vector")
  # TODO: check all metas the same, all structures same, etc.
  meta = first(models).meta
  VectorBatchNLPModel{eltype(meta.x0), typeof(meta.x0), M}(models, Counters(), meta)
end
Base.length(vnlp::VectorBatchNLPModel) = length(vnlp.models)
Base.getindex(vnlp::VectorBatchNLPModel, i::Integer) = vnlp.models[i]
Base.iterate(vnlp::VectorBatchNLPModel, state::Integer = 1) = iterate(vnlp.models, state)

function _batch_map(f, bnlp::VectorBatchNLPModel, xs::VV...)
  n = length(bnlp)
  results = Vector{Any}(undef, n)
  for i = 1:n
    args_i = (x[i] for x in xs)
    results[i] = f(bnlp[i], args_i...)
  end
  return results
end

function _batch_map!(f, bnlp::VectorBatchNLPModel, outputs::Vector, xs::VV...)
  n = length(bnlp)
  for i = 1:n
    args_i = (x[i] for x in xs)
    f(bnlp[i], outputs[i], args_i...)
  end
  return outputs
end

function _batch_map_weight(f, bnlp::VectorBatchNLPModel, obj_weights::Vector, xs::VV...)
  n = length(bnlp)
  results = Vector{Any}(undef, n)
  for i = 1:n
    args_i = (x[i] for x in xs)
    results[i] = f(bnlp[i], args_i...; obj_weight = obj_weights[i])
  end
  return results
end

function _batch_map_weight!(
  f,
  bnlp::VectorBatchNLPModel,
  outputs::Vector,
  obj_weights::Vector,
  xs::VV...,
)
  n = length(bnlp)
  for i = 1:n
    args_i = (x[i] for x in xs)
    f(bnlp[i], outputs[i], args_i...; obj_weight = obj_weights[i])
  end
  return outputs
end

function _batch_map_tuple(f, bnlp::VectorBatchNLPModel, xs::VV...)
  n = length(bnlp)
  results = _batch_map(f, bnlp, xs...)
  # Get types from first result
  first_result = results[1]
  T1 = typeof(first_result[1])
  T2 = typeof(first_result[2])
  vec1 = Vector{T1}(undef, n)
  vec2 = Vector{T2}(undef, n)
  for i = 1:n
    vec1[i], vec2[i] = results[i]
  end
  return vec1, vec2
end

function _batch_map_tuple!(f, bnlp::VectorBatchNLPModel, outputs::Vector, xs::VV...)
  n = length(bnlp)
  firsts = Vector{eltype(bnlp.meta.x0)}(undef, n)
  for i = 1:n
    args_i = (x[i] for x in xs)
    firsts[i], _ = f(bnlp[i], args_i..., outputs[i])
  end
  return firsts, outputs
end

for fun in fieldnames(Counters)
  @eval function NLPModels.increment!(bnlp::VectorBatchNLPModel, ::Val{$(Meta.quot(fun))})
    # sub-model counters are already incremented since we call their methods
    bnlp.counters.$fun += 1
  end
end