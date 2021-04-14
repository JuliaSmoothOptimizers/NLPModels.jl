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
mutable struct SimpleNLPModel <: AbstractNLPModel
  meta::NLPModelMeta
  counters::Counters
end

function SimpleNLPModel()
  meta = NLPModelMeta(
    2,
    nnzh = 2,
    ncon = 2,
    lvar = zeros(2),
    uvar = ones(2),
    x0 = [2.0; 2.0],
    lcon = [0.0; 0.0],
    ucon = [0.0; Inf],
    name = "Simple NLP Model",
  )

  return SimpleNLPModel(meta, Counters())
end

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
  vals[1] -= y[2] / 2
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
  Hv[1] -= y[2] * v[1] / 2
  Hv[2] -= 2y[2] * v[2]
  return Hv
end

function NLPModels.cons!(nlp::SimpleNLPModel, x::AbstractVector, cx::AbstractVector)
  @lencheck 2 x cx
  increment!(nlp, :neval_cons)
  cx .= [x[1] - 2 * x[2] + 1; -x[1]^2 / 4 - x[2]^2 + 1]
  return cx
end

function NLPModels.jac_structure!(
  nlp::SimpleNLPModel,
  rows::AbstractVector{Int},
  cols::AbstractVector{Int},
)
  @lencheck 4 rows cols
  rows .= [1, 2, 1, 2]
  cols .= [1, 1, 2, 2]
  return rows, cols
end

function NLPModels.jac_coord!(nlp::SimpleNLPModel, x::AbstractVector, vals::AbstractVector)
  @lencheck 2 x
  @lencheck 4 vals
  increment!(nlp, :neval_jac)
  vals .= [1, -x[1] / 2, -2, -2 * x[2]]
  return vals
end

function NLPModels.jprod!(
  nlp::SimpleNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck 2 x v Jv
  increment!(nlp, :neval_jprod)
  Jv .= [v[1] - 2 * v[2]; -x[1] * v[1] / 2 - 2 * x[2] * v[2]]
  return Jv
end

function NLPModels.jtprod!(
  nlp::SimpleNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck 2 x v Jtv
  increment!(nlp, :neval_jtprod)
  Jtv .= [v[1] - x[1] * v[2] / 2; -2 * v[1] - 2 * x[2] * v[2]]
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
    vals[1] = -1 / 2
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
    Hv[1] = -v[1] / 2
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
  gHv .= [T(0); -g[1] * v[1] / 2 - 2 * g[2] * v[2]]
  return gHv
end
