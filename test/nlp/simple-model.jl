"""
    SimpleNLPModel <: AbstractNLPModel

Simple model for testing purposes.
Modified problem 14 in the Hock-Schittkowski suite

     min   (x₁ - 2)² + (x₂ - 1)²
    s.to   x₁ - 2x₂ + 1 = 0
           -x₁² / 4 - x₂² + 1 ≥ 0
           0 ≤ x ≤ 1

x₀ = [2.0, 2.0].
"""
mutable struct SimpleNLPModel{T, S} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  p::T
end

function SimpleNLPModel(::Type{T}; p = T(4)) where {T}
  meta = NLPModelMeta(
    2,
    nnzh = 2,
    ncon = 2,
    lvar = zeros(T, 2),
    uvar = ones(T, 2),
    x0 = [T(2.0); T(2.0)],
    lcon = [zero(T); zero(T)],
    ucon = [zero(T); T(Inf)],
    name = "Simple NLP Model",
    lin = [1],
    lin_nnzj = 2,
    nln_nnzj = 2,
  )

  return SimpleNLPModel(meta, Counters(), T(p))
end

SimpleNLPModel(; p = 4.0) = SimpleNLPModel(Float64; p = p)

function NLPModels.obj(nlp::SimpleNLPModel, x::AbstractVector)
  @lencheck 2 x
  increment!(nlp, :neval_obj)
  return (x[1] - 2)^2 + (x[2] - 1)^2
end

function NLPModels.grad!(nlp::SimpleNLPModel, x::AbstractVector, gx::AbstractVector)
  @lencheck 2 x gx
  increment!(nlp, :neval_grad)
  gx .= [2 * (x[1] - 2); 2 * (x[2] - 1)]
  return gx
end

function NLPModels.hess_structure!(
  nlp::SimpleNLPModel,
  rows::AbstractVector{Int},
  cols::AbstractVector{Int},
)
  @lencheck 2 rows cols
  rows[1] = 1
  rows[2] = 2
  cols[1] = 1
  cols[2] = 2
  return rows, cols
end

function NLPModels.hess_coord!(
  nlp::SimpleNLPModel,
  x::AbstractVector{T},
  y::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x y vals
  increment!(nlp, :neval_hess)
  vals .= 2obj_weight
  vals[1] -= 2y[2] / nlp.p
  vals[2] -= 2y[2]
  return vals
end

function NLPModels.hprod!(
  nlp::SimpleNLPModel,
  x::AbstractVector{T},
  y::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x y v Hv
  increment!(nlp, :neval_hprod)
  Hv .= 2obj_weight * v
  Hv[1] -= (2y[2] / nlp.p) * v[1]
  Hv[2] -= 2y[2] * v[2]
  return Hv
end

function NLPModels.cons_nln!(nlp::SimpleNLPModel, x::AbstractVector, cx::AbstractVector)
  @lencheck 2 x
  @lencheck 1 cx
  increment!(nlp, :neval_cons_nln)
  cx .= [-x[1]^2 / nlp.p - x[2]^2 + 1]
  return cx
end

function NLPModels.cons_lin!(nlp::SimpleNLPModel, x::AbstractVector, cx::AbstractVector)
  @lencheck 2 x
  @lencheck 1 cx
  increment!(nlp, :neval_cons_lin)
  cx .= [x[1] - 2 * x[2] + 1]
  return cx
end

function NLPModels.jac_nln_structure!(
  nlp::SimpleNLPModel,
  rows::AbstractVector{Int},
  cols::AbstractVector{Int},
)
  @lencheck 2 rows cols
  rows .= [1, 1]
  cols .= [1, 2]
  return rows, cols
end

function NLPModels.jac_lin_structure!(
  nlp::SimpleNLPModel,
  rows::AbstractVector{Int},
  cols::AbstractVector{Int},
)
  @lencheck 2 rows cols
  rows .= [1, 1]
  cols .= [1, 2]
  return rows, cols
end

function NLPModels.jac_nln_coord!(nlp::SimpleNLPModel, x::AbstractVector, vals::AbstractVector)
  @lencheck 2 x vals
  increment!(nlp, :neval_jac_nln)
  vals .= [-2 * x[1] / nlp.p, -2 * x[2]]
  return vals
end

function NLPModels.jac_lin_coord!(nlp::SimpleNLPModel, x::AbstractVector, vals::AbstractVector)
  @lencheck 2 x vals
  increment!(nlp, :neval_jac_lin)
  vals .= [1, -2]
  return vals
end

function NLPModels.jprod_nln!(
  nlp::SimpleNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck 2 x v
  @lencheck 1 Jv
  increment!(nlp, :neval_jprod_nln)
  Jv .= [-(2 * x[1] / nlp.p) * v[1] - 2 * x[2] * v[2]]
  return Jv
end

function NLPModels.jprod_lin!(
  nlp::SimpleNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck 2 x v
  @lencheck 1 Jv
  increment!(nlp, :neval_jprod_lin)
  Jv .= [v[1] - 2 * v[2]]
  return Jv
end

function NLPModels.jtprod_nln!(
  nlp::SimpleNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck 2 x Jtv
  @lencheck 1 v
  increment!(nlp, :neval_jtprod_nln)
  Jtv .= [-(2 * x[1] / nlp.p) * v[1]; -2 * x[2] * v[1]]
  return Jtv
end

function NLPModels.jtprod_lin!(
  nlp::SimpleNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck 2 x Jtv
  @lencheck 1 v
  increment!(nlp, :neval_jtprod_lin)
  Jtv .= [v[1]; -2 * v[1]]
  return Jtv
end

function NLPModels.jth_hess_coord!(
  nlp::SimpleNLPModel,
  x::AbstractVector{T},
  j::Integer,
  vals::AbstractVector{T},
) where {T}
  @lencheck 2 x vals
  increment!(nlp, :neval_jhess)
  if j == 1
    vals .= 0
  elseif j == 2
    vals[1] = -2 / nlp.p
    vals[2] = -2
  end
  return vals
end

function NLPModels.jth_hprod!(
  nlp::SimpleNLPModel,
  x::AbstractVector{T},
  v::AbstractVector{T},
  j::Integer,
  Hv::AbstractVector{T},
) where {T}
  @lencheck 2 x v Hv
  increment!(nlp, :neval_jhprod)
  if j == 1
    Hv .= 0
  elseif j == 2
    Hv[1] = -(2 / nlp.p) * v[1]
    Hv[2] = -2v[2]
  end
  return Hv
end

function NLPModels.ghjvprod!(
  nlp::SimpleNLPModel,
  x::AbstractVector{T},
  g::AbstractVector{T},
  v::AbstractVector{T},
  gHv::AbstractVector{T},
) where {T}
  @lencheck nlp.meta.nvar x g v
  @lencheck nlp.meta.ncon gHv
  increment!(nlp, :neval_hprod)
  gHv .= [T(0); -(2 * g[1] / nlp.p) * v[1] - 2 * g[2] * v[2]]
  return gHv
end

mutable struct BatchSimpleNLPModel{T,S,VI} <: AbstractBatchNLPModel{T,S,VI}
    meta::BatchNLPModelMeta{T,S,VI}
    models::Vector{SimpleNLPModel{T,S}}
end

function BatchSimpleNLPModel(ps::Vector{T}) where T
    return BatchSimpleNLPModel(
        BatchNLPModelMeta{T, Vector{T}, Vector{Int}}(
            length(ps),
            2;
            nnzh = 2,
            ncon = 2,
            lvar = zeros(T, 2),
            uvar = ones(T, 2),
            x0 = [T(2.0); T(2.0)],
            lcon = [zero(T); zero(T)],
            ucon = [zero(T); T(Inf)],
            name = "Simple NLP Model",
            # lin = [1],
            # lin_nnzj = 2,
            # nln_nnzj = 2,
            nnzj=4,
        ),
        [SimpleNLPModel(; p = p) for p in ps]
    )
end

function _slice(batch_buffer, batch_idx, single_length)
    # batch_buffer[(batch_idx - 1) * single_length : batch_idx * single_length]
    unsafe_wrap(typeof(batch_buffer), pointer(batch_buffer, (batch_idx - 1) * single_length + 1), single_length)
end

function NLPModels.batch_obj!(bnlp::BatchSimpleNLPModel{T,S,VI}, bx, bf) where {T,S,VI}
    for (i, nlp) in enumerate(bnlp.models)
        bxi = _slice(bx, i, bnlp.meta.nvar)
        bf[i] = NLPModels.obj(nlp, bxi)
    end
    return bf
end

function NLPModels.batch_grad!(bnlp::BatchSimpleNLPModel{T,S,VI}, bx, bg) where {T,S,VI}
    for (i, nlp) in enumerate(bnlp.models)
        bxi = _slice(bx, i, bnlp.meta.nvar)
        bgi = _slice(bg, i, bnlp.meta.nvar)
        NLPModels.grad!(nlp, bxi, bgi)
    end
    return bg
end

function NLPModels.batch_cons!(bnlp::BatchSimpleNLPModel{T,S,VI}, bx, bc) where {T,S,VI}
    for (i, nlp) in enumerate(bnlp.models)
        bxi = _slice(bx, i, bnlp.meta.nvar)
        bci = _slice(bc, i, bnlp.meta.ncon)
        NLPModels.cons!(nlp, bxi, bci)
    end
    return bc
end

NLPModels.batch_jac_structure!(bnlp, jrows, jcols) = NLPModels.jac_structure!(bnlp.models[1], jrows, jcols)

function NLPModels.batch_jac_coord!(bnlp::BatchSimpleNLPModel{T,S,VI}, bx, bjvals) where {T,S,VI}
    for (i, nlp) in enumerate(bnlp.models)
        bxi = _slice(bx, i, bnlp.meta.nvar)
        bjvalsi = _slice(bjvals, i, bnlp.meta.nnzj)
        NLPModels.jac_coord!(nlp, bxi, bjvalsi)
    end
    return bjvals
end

function NLPModels.batch_jprod!(bnlp::BatchSimpleNLPModel{T,S,VI}, bx, bv, bJv) where {T,S,VI}
    for (i, nlp) in enumerate(bnlp.models)
        bxi = _slice(bx, i, bnlp.meta.nvar)
        bvi = _slice(bv, i, bnlp.meta.nvar)
        bJvi = _slice(bJv, i, bnlp.meta.ncon)
        NLPModels.jprod!(nlp, bxi, bvi, bJvi)
    end
    return bJv
end

function NLPModels.batch_jtprod!(bnlp::BatchSimpleNLPModel{T,S,VI}, bx, bv, bJtv) where {T,S,VI}
    for (i, nlp) in enumerate(bnlp.models)
        bxi = _slice(bx, i, bnlp.meta.nvar)
        bvi = _slice(bv, i, bnlp.meta.ncon)
        bJtvi = _slice(bJtv, i, bnlp.meta.nvar)
        NLPModels.jtprod!(nlp, bxi, bvi, bJtvi)
    end
    return bJtv
end

NLPModels.batch_hess_structure!(bnlp, jrows, jcols) = NLPModels.hess_structure!(bnlp.models[1], jrows, jcols)


function NLPModels.batch_hess_coord!(bnlp::BatchSimpleNLPModel{T,S,VI}, bx, by, bobj_weight, bhvals) where {T,S,VI}
    for (i, nlp) in enumerate(bnlp.models)
        bxi = _slice(bx, i, bnlp.meta.nvar)
        byi = _slice(by, i, bnlp.meta.ncon)
        bobj_weighti = bobj_weight[i]
        bhvalsi = _slice(bhvals, i, bnlp.meta.nnzh)
        NLPModels.hess_coord!(nlp, bxi, byi, bhvalsi; obj_weight=bobj_weighti)
    end
    return bhvals
end

function NLPModels.batch_hprod!(bnlp::BatchSimpleNLPModel{T,S,VI}, bx, by, bv, bobj_weight, bHv) where {T,S,VI}
    for (i, nlp) in enumerate(bnlp.models)
        bxi = _slice(bx, i, bnlp.meta.nvar)
        byi = _slice(by, i, bnlp.meta.ncon)
        bvi = _slice(bv, i, bnlp.meta.nvar)
        bobj_weighti = bobj_weight[i]
        bHvi = _slice(bHv, i, bnlp.meta.nvar)
        NLPModels.hprod!(nlp, bxi, byi, bvi, bHvi; obj_weight = bobj_weighti)
    end
    return bHv
end
