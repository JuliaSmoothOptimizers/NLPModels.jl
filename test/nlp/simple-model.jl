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

mutable struct BatchSimpleNLPModel{T, M, S} <: AbstractBatchNLPModel{T, M}
  meta::BatchNLPModelMeta{T, M}
  models::Vector{SimpleNLPModel{T, S}}
end

function BatchSimpleNLPModel(ps::Vector{T}) where {T}
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
    [SimpleNLPModel(; p = p) for p in ps],
  )
end

function NLPModels.obj!(bnlp::BatchSimpleNLPModel, bx, bf)
  for (i, nlp) in enumerate(bnlp.models)
    bf[i] = NLPModels.obj(nlp, view(bx, :, i))
  end
  return bf
end

function NLPModels.grad!(bnlp::BatchSimpleNLPModel, bx, bg)
  for (i, nlp) in enumerate(bnlp.models)
    NLPModels.grad!(nlp, view(bx, :, i), view(bg, :, i))
  end
  return bg
end

function NLPModels.cons!(bnlp::BatchSimpleNLPModel, bx, bc)
  for (i, nlp) in enumerate(bnlp.models)
    NLPModels.cons!(nlp, view(bx, :, i), view(bc, :, i))
  end
  return bc
end

NLPModels.jac_structure!(bnlp::BatchSimpleNLPModel, jrows, jcols) =
  NLPModels.jac_structure!(bnlp.models[1], jrows, jcols)

function NLPModels.jac_coord!(bnlp::BatchSimpleNLPModel, bx, bjvals)
  for (i, nlp) in enumerate(bnlp.models)
    NLPModels.jac_coord!(nlp, view(bx, :, i), view(bjvals, :, i))
  end
  return bjvals
end

function NLPModels.jprod!(bnlp::BatchSimpleNLPModel, bx, bv, bJv)
  for (i, nlp) in enumerate(bnlp.models)
    NLPModels.jprod!(nlp, view(bx, :, i), view(bv, :, i), view(bJv, :, i))
  end
  return bJv
end

function NLPModels.jtprod!(bnlp::BatchSimpleNLPModel, bx, bv, bJtv)
  for (i, nlp) in enumerate(bnlp.models)
    NLPModels.jtprod!(nlp, view(bx, :, i), view(bv, :, i), view(bJtv, :, i))
  end
  return bJtv
end

NLPModels.hess_structure!(bnlp::BatchSimpleNLPModel, jrows, jcols) =
  NLPModels.hess_structure!(bnlp.models[1], jrows, jcols)

function NLPModels.hess_coord!(bnlp::BatchSimpleNLPModel, bx, by, bobj_weight, bhvals)
  for (i, nlp) in enumerate(bnlp.models)
    NLPModels.hess_coord!(
      nlp,
      view(bx, :, i),
      view(by, :, i),
      view(bhvals, :, i);
      obj_weight = bobj_weight[i],
    )
  end
  return bhvals
end

function NLPModels.hprod!(bnlp::BatchSimpleNLPModel, bx, by, bv, bobj_weight, bHv)
  for (i, nlp) in enumerate(bnlp.models)
    NLPModels.hprod!(
      nlp,
      view(bx, :, i),
      view(by, :, i),
      view(bv, :, i),
      view(bHv, :, i);
      obj_weight = bobj_weight[i],
    )
  end
  return bHv
end

"""
    SimpleParamNLPModel <: AbstractNLPModel

Simple parametric model for testing the parametric NLP API.

    min   f(x, p) = (x₁ - p₁)² + (x₂ - p₂)²
    s.t.  c(x, p) = x₁ + p₁*x₂    with  -p₂ ≤ c(x,p) ≤ p₂
          -p₁ ≤ x₁ ≤ p₁
          -p₂ ≤ x₂ ≤ p₂

x₀ = [0, 0], p = [p₁, p₂].
"""
mutable struct SimpleParamNLPModel{T, S} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  ps::S
end

function SimpleParamNLPModel(::Type{T}; ps = T[2, 3]) where {T}
  p1, p2 = ps[1], ps[2]
  meta = NLPModelMeta(
    2,
    ncon = 1,
    x0 = zeros(T, 2),
    lvar = T[-p1, -p2],
    uvar = T[p1, p2],
    lcon = T[-p2],
    ucon = T[p2],
    nnzh = 2,
    nnzj = 2,
    name = "Simple Parametric NLP Model",
    nparam = 2,
    nnzjp = 1,
    nnzhp = 3,
    nnzgp = 2,
    nnzjplcon = 1,
    nnzjpucon = 1,
    nnzjplvar = 2,
    nnzjpuvar = 2,
    grad_param_available = true,
    jac_param_available = true,
    hess_param_available = true,
    jpprod_available = true,
    jptprod_available = true,
    hpprod_available = true,
    hptprod_available = true,
    lcon_jac_available = true,
    ucon_jac_available = true,
    lvar_jac_available = true,
    uvar_jac_available = true,
    lcon_jpprod_available = true,
    ucon_jpprod_available = true,
    lvar_jpprod_available = true,
    uvar_jpprod_available = true,
    lcon_jptprod_available = true,
    ucon_jptprod_available = true,
    lvar_jptprod_available = true,
    uvar_jptprod_available = true,
  )
  return SimpleParamNLPModel(meta, Counters(), T.(ps))
end

SimpleParamNLPModel(; ps = [2.0, 3.0]) = SimpleParamNLPModel(Float64; ps = ps)

function NLPModels.obj(nlp::SimpleParamNLPModel, x::AbstractVector)
  @lencheck 2 x
  increment!(nlp, :neval_obj)
  p1, p2 = nlp.ps
  return (x[1] - p1)^2 + (x[2] - p2)^2
end

function NLPModels.grad!(nlp::SimpleParamNLPModel, x::AbstractVector, g::AbstractVector)
  @lencheck 2 x g
  increment!(nlp, :neval_grad)
  p1, p2 = nlp.ps
  g .= [2 * (x[1] - p1); 2 * (x[2] - p2)]
  return g
end

function NLPModels.cons!(nlp::SimpleParamNLPModel, x::AbstractVector, c::AbstractVector)
  @lencheck 2 x
  @lencheck 1 c
  increment!(nlp, :neval_cons)
  p1 = nlp.ps[1]
  c[1] = x[1] + p1 * x[2]
  return c
end

function NLPModels.jac_structure!(nlp::SimpleParamNLPModel, rows::AbstractVector{Int}, cols::AbstractVector{Int})
  @lencheck 2 rows cols
  rows .= [1, 1]
  cols .= [1, 2]
  return rows, cols
end

function NLPModels.jac_coord!(nlp::SimpleParamNLPModel, x::AbstractVector, vals::AbstractVector)
  @lencheck 2 x vals
  increment!(nlp, :neval_jac)
  p1 = nlp.ps[1]
  vals .= [1, p1]
  return vals
end

function NLPModels.hess_structure!(nlp::SimpleParamNLPModel, rows::AbstractVector{Int}, cols::AbstractVector{Int})
  @lencheck 2 rows cols
  rows .= [1, 2]
  cols .= [1, 2]
  return rows, cols
end

function NLPModels.hess_coord!(
  nlp::SimpleParamNLPModel,
  x::AbstractVector{T},
  y::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x y vals
  increment!(nlp, :neval_hess)
  vals .= 2 * obj_weight
  return vals
end

# Parametric API

function NLPModels.grad_param!(nlp::SimpleParamNLPModel, x::AbstractVector, g::AbstractVector)
  @lencheck 2 x g
  p1, p2 = nlp.ps
  g .= [-2 * (x[1] - p1); -2 * (x[2] - p2)]
  return g
end

function NLPModels.jac_param_structure!(nlp::SimpleParamNLPModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
  @lencheck 1 rows cols
  rows[1] = 1
  cols[1] = 1
  return rows, cols
end

function NLPModels.jac_param_coord!(nlp::SimpleParamNLPModel, x::AbstractVector, vals::AbstractVector)
  @lencheck 2 x
  @lencheck 1 vals
  vals[1] = x[2]
  return vals
end

function NLPModels.jpprod!(
  nlp::SimpleParamNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck 2 x v
  @lencheck 1 Jv
  Jv[1] = x[2] * v[1]
  return Jv
end

function NLPModels.jptprod!(
  nlp::SimpleParamNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck 2 x Jtv
  @lencheck 1 v
  Jtv .= [x[2] * v[1]; 0]
  return Jtv
end

function NLPModels.hess_param_structure!(nlp::SimpleParamNLPModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
  @lencheck 3 rows cols
  rows .= [1, 2, 2]
  cols .= [1, 1, 2]
  return rows, cols
end

function NLPModels.hess_param_coord!(
  nlp::SimpleParamNLPModel,
  x::AbstractVector{T},
  y::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x
  @lencheck 1 y
  @lencheck 3 vals
  σ = T(obj_weight)
  vals .= [-2σ, y[1], -2σ]
  return vals
end

function NLPModels.hpprod!(
  nlp::SimpleParamNLPModel,
  x::AbstractVector{T},
  y::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x v Hv
  @lencheck 1 y
  σ = T(obj_weight)
  Hv .= [-2σ * v[1]; y[1] * v[1] - 2σ * v[2]]
  return Hv
end

function NLPModels.hptprod!(
  nlp::SimpleParamNLPModel,
  x::AbstractVector{T},
  y::AbstractVector{T},
  v::AbstractVector{T},
  Htv::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x v Htv
  @lencheck 1 y
  σ = T(obj_weight)
  Htv .= [-2σ * v[1] + y[1] * v[2]; -2σ * v[2]]
  return Htv
end

function NLPModels.lcon_jac_param_structure!(nlp::SimpleParamNLPModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
  @lencheck 1 rows cols
  rows[1] = 1
  cols[1] = 2
  return rows, cols
end

function NLPModels.lcon_jac_param_coord!(nlp::SimpleParamNLPModel, vals::AbstractVector)
  @lencheck 1 vals
  vals[1] = -1
  return vals
end

function NLPModels.lcon_jpprod!(nlp::SimpleParamNLPModel, v::AbstractVector, Jv::AbstractVector)
  @lencheck 2 v
  @lencheck 1 Jv
  Jv[1] = -v[2]
  return Jv
end

function NLPModels.lcon_jptprod!(nlp::SimpleParamNLPModel, v::AbstractVector, Jtv::AbstractVector)
  @lencheck 1 v
  @lencheck 2 Jtv
  Jtv .= [0; -v[1]]
  return Jtv
end

function NLPModels.ucon_jac_param_structure!(nlp::SimpleParamNLPModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
  @lencheck 1 rows cols
  rows[1] = 1
  cols[1] = 2
  return rows, cols
end

function NLPModels.ucon_jac_param_coord!(nlp::SimpleParamNLPModel, vals::AbstractVector)
  @lencheck 1 vals
  vals[1] = 1
  return vals
end

function NLPModels.ucon_jpprod!(nlp::SimpleParamNLPModel, v::AbstractVector, Jv::AbstractVector)
  @lencheck 2 v
  @lencheck 1 Jv
  Jv[1] = v[2]
  return Jv
end

function NLPModels.ucon_jptprod!(nlp::SimpleParamNLPModel, v::AbstractVector, Jtv::AbstractVector)
  @lencheck 1 v
  @lencheck 2 Jtv
  Jtv .= [0; v[1]]
  return Jtv
end

function NLPModels.lvar_jac_param_structure!(nlp::SimpleParamNLPModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
  @lencheck 2 rows cols
  rows .= [1, 2]
  cols .= [1, 2]
  return rows, cols
end

function NLPModels.lvar_jac_param_coord!(nlp::SimpleParamNLPModel, vals::AbstractVector)
  @lencheck 2 vals
  vals .= [-1, -1]
  return vals
end

function NLPModels.lvar_jpprod!(nlp::SimpleParamNLPModel, v::AbstractVector, Jv::AbstractVector)
  @lencheck 2 v Jv
  Jv .= -v
  return Jv
end

function NLPModels.lvar_jptprod!(nlp::SimpleParamNLPModel, v::AbstractVector, Jtv::AbstractVector)
  @lencheck 2 v Jtv
  Jtv .= -v
  return Jtv
end

function NLPModels.uvar_jac_param_structure!(nlp::SimpleParamNLPModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
  @lencheck 2 rows cols
  rows .= [1, 2]
  cols .= [1, 2]
  return rows, cols
end

function NLPModels.uvar_jac_param_coord!(nlp::SimpleParamNLPModel, vals::AbstractVector)
  @lencheck 2 vals
  vals .= [1, 1]
  return vals
end

function NLPModels.uvar_jpprod!(nlp::SimpleParamNLPModel, v::AbstractVector, Jv::AbstractVector)
  @lencheck 2 v Jv
  Jv .= v
  return Jv
end

function NLPModels.uvar_jptprod!(nlp::SimpleParamNLPModel, v::AbstractVector, Jtv::AbstractVector)
  @lencheck 2 v Jtv
  Jtv .= v
  return Jtv
end
