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

mutable struct BatchSimpleNLPModel{T,M,S} <: AbstractBatchNLPModel{T,M}
    meta::BatchNLPModelMeta{T,M}
    models::Vector{SimpleNLPModel{T,S}}
end

function BatchSimpleNLPModel(ps::Vector{T}) where T
    return BatchSimpleNLPModel(
        BatchNLPModelMeta{T, Matrix{T}}(
            length(ps),
            2;
            nnzh = 2,
            ncon = 2,
            lvar = zeros(T, 1, 2),
            uvar = ones(T, 1, 2),
            x0 = [T(2.0) T(2.0);],
            lcon = [zero(T) zero(T);],
            ucon = [zero(T) T(Inf);],
            name = "Batch simple NLP Model",
            nnzj = 4,
        ),
        [SimpleNLPModel(; p = p) for p in ps]
    )
end

function NLPModels.obj!(bnlp::BatchSimpleNLPModel, bx, bf)
    for (i, nlp) in enumerate(bnlp.models)
        bf[i] = NLPModels.obj(nlp, view(bx,:,i))
    end
    return bf
end

function NLPModels.grad!(bnlp::BatchSimpleNLPModel, bx, bg)
    for (i, nlp) in enumerate(bnlp.models)
        NLPModels.grad!(nlp, view(bx,:,i), view(bg,:,i))
    end
    return bg
end

function NLPModels.cons!(bnlp::BatchSimpleNLPModel, bx, bc)
    for (i, nlp) in enumerate(bnlp.models)
        NLPModels.cons!(nlp, view(bx,:,i), view(bc,:,i))
    end
    return bc
end

NLPModels.jac_structure!(bnlp::BatchSimpleNLPModel, jrows, jcols) = NLPModels.jac_structure!(bnlp.models[1], jrows, jcols)

function NLPModels.jac_coord!(bnlp::BatchSimpleNLPModel, bx, bjvals)
    for (i, nlp) in enumerate(bnlp.models)
        NLPModels.jac_coord!(nlp, view(bx,:,i), view(bjvals,:,i))
    end
    return bjvals
end

function NLPModels.jprod!(bnlp::BatchSimpleNLPModel, bx, bv, bJv)
    for (i, nlp) in enumerate(bnlp.models)
        NLPModels.jprod!(nlp, view(bx,:,i), view(bv,:,i), view(bJv,:,i))
    end
    return bJv
end

function NLPModels.jtprod!(bnlp::BatchSimpleNLPModel, bx, bv, bJtv)
    for (i, nlp) in enumerate(bnlp.models)
        NLPModels.jtprod!(nlp, view(bx,:,i), view(bv,:,i), view(bJtv,:,i))
    end
    return bJtv
end

NLPModels.hess_structure!(bnlp::BatchSimpleNLPModel, jrows, jcols) = NLPModels.hess_structure!(bnlp.models[1], jrows, jcols)


function NLPModels.hess_coord!(bnlp::BatchSimpleNLPModel, bx, by, bobj_weight, bhvals)
    for (i, nlp) in enumerate(bnlp.models)
        NLPModels.hess_coord!(nlp, view(bx,:,i), view(by,:,i), view(bhvals,:,i); obj_weight = bobj_weight[i])
    end
    return bhvals
end

function NLPModels.hprod!(bnlp::BatchSimpleNLPModel, bx, by, bv, bobj_weight, bHv)
    for (i, nlp) in enumerate(bnlp.models)
        NLPModels.hprod!(nlp, view(bx,:,i), view(by,:,i), view(bv,:,i), view(bHv,:,i); obj_weight = bobj_weight[i])
    end
    return bHv
end
