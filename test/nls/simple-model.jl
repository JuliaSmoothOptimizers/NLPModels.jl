"""
    SimpleNLSModel <: AbstractNLSModel

Simple NLSModel for testing purposes.
Modified problem 20 in the Hock-Schittkowski Suite.

     min   ½‖F(x)‖²
    s.to   x₁ + x₂² ≥ 0
           x₁² + x₂ ≥ 0
           x₁² + x₂² = 1
           0 ≤ x ≤ 1,

where

    F(x) = [1 - x₁; 10 (x₂ - x₁²)]

x₀ = ones(n).

Modified SimpleNLSModel.
"""
mutable struct SimpleNLSModel <: AbstractNLSModel
  meta::NLPModelMeta
  nls_meta::NLSMeta
  counters::NLSCounters
end

function SimpleNLSModel()
  meta = NLPModelMeta(
    2,
    x0 = ones(2),
    name = "Simple NLS Model",
    lvar = zeros(2),
    uvar = ones(2),
    ncon = 3,
    lcon = [0.0; 0.0; 1.0],
    ucon = [Inf; Inf; 1.0],
    nnzj = 6,
  )
  nls_meta = NLSMeta(2, 2, nnzj = 3, nnzh = 1)

  return SimpleNLSModel(meta, nls_meta, NLSCounters())
end

function NLPModels.residual!(nls::SimpleNLSModel, x::AbstractVector, Fx::AbstractVector)
  @lencheck 2 x Fx
  increment!(nls, :neval_residual)
  Fx .= [1 - x[1]; 10 * (x[2] - x[1]^2)]
  return Fx
end

# Jx = [-1  0; -20x₁  10]
function NLPModels.jac_structure_residual!(
  nls::SimpleNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck 3 rows cols
  rows .= [1, 2, 2]
  cols .= [1, 1, 2]
  return rows, cols
end

function NLPModels.jac_coord_residual!(nls::SimpleNLSModel, x::AbstractVector, vals::AbstractVector)
  @lencheck 2 x
  @lencheck 3 vals
  increment!(nls, :neval_jac_residual)
  vals .= [-1, -20x[1], 10]
  return vals
end

function NLPModels.jprod_residual!(
  nls::SimpleNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck 2 x v Jv
  increment!(nls, :neval_jprod_residual)
  Jv .= [-v[1]; -20 * x[1] * v[1] + 10 * v[2]]
  return Jv
end

function NLPModels.jtprod_residual!(
  nls::SimpleNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck 2 x v Jtv
  increment!(nls, :neval_jtprod_residual)
  Jtv .= [-v[1] - 20 * x[1] * v[2]; 10 * v[2]]
  return Jtv
end

function NLPModels.hess_structure_residual!(
  nls::SimpleNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck 1 rows cols
  rows[1] = 1
  cols[1] = 1
  return rows, cols
end

function NLPModels.hess_coord_residual!(
  nls::SimpleNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  vals::AbstractVector,
)
  @lencheck 2 x v
  @lencheck 1 vals
  increment!(nls, :neval_hess_residual)
  vals[1] = -20v[2]
  return vals
end

function NLPModels.hprod_residual!(
  nls::SimpleNLSModel,
  x::AbstractVector,
  i::Int,
  v::AbstractVector,
  Hiv::AbstractVector,
)
  @lencheck 2 x v Hiv
  increment!(nls, :neval_hprod_residual)
  if i == 2
    Hiv .= [-20v[1]; 0]
  else
    Hiv .= zero(eltype(x))
  end
  return Hiv
end

function NLPModels.cons!(nls::SimpleNLSModel, x::AbstractVector, cx::AbstractVector)
  @lencheck 2 x
  @lencheck 3 cx
  increment!(nls, :neval_cons)
  cx .= [x[1] + x[2]^2; x[1]^2 + x[2]; x[1]^2 + x[2]^2 - 1]
  return cx
end

function NLPModels.jac_structure!(
  nls::SimpleNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck 6 rows cols
  rows .= [1, 1, 2, 2, 3, 3]
  cols .= [1, 2, 1, 2, 1, 2]
  return rows, cols
end

function NLPModels.jac_coord!(nls::SimpleNLSModel, x::AbstractVector, vals::AbstractVector)
  @lencheck 2 x
  @lencheck 6 vals
  increment!(nls, :neval_jac)
  vals .= [1, 2x[2], 2x[1], 1, 2x[1], 2x[2]]
  return vals
end

function NLPModels.jprod!(
  nls::SimpleNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck 2 x v
  @lencheck 3 Jv
  increment!(nls, :neval_jprod)
  Jv .= [v[1] + 2x[2] * v[2]; 2x[1] * v[1] + v[2]; 2x[1] * v[1] + 2x[2] * v[2]]
  return Jv
end

function NLPModels.jtprod!(
  nls::SimpleNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck 2 x Jtv
  @lencheck 3 v
  increment!(nls, :neval_jtprod)
  Jtv .= [v[1] + 2x[1] * (v[2] + v[3]); v[2] + 2x[2] * (v[1] + v[3])]
  return Jtv
end

function NLPModels.hess(nls::SimpleNLSModel, x::AbstractVector{T}; obj_weight = 1.0) where {T}
  @lencheck 2 x
  increment!(nls, :neval_hess)
  return obj_weight * [T(1) - 200 * x[2]+600 * x[1]^2 T(0); -200*x[1] T(100)]
end

function NLPModels.hess(
  nls::SimpleNLSModel,
  x::AbstractVector{T},
  y::AbstractVector{T};
  obj_weight = 1.0,
) where {T}
  @lencheck 2 x
  @lencheck 3 y
  increment!(nls, :neval_hess)
  return [
    obj_weight*(T(1) - 200 * x[2] + 600 * x[1]^2)+2*y[2]+2*y[3] T(0)
    -obj_weight*200*x[1] obj_weight*T(100)+2*y[1]+2*y[3]
  ]
end

function NLPModels.hess_structure!(
  nls::SimpleNLSModel,
  rows::AbstractVector{Int},
  cols::AbstractVector{Int},
)
  @lencheck 3 rows cols
  n = nls.meta.nvar
  I = ((i, j) for i = 1:n, j = 1:n if i ≥ j)
  rows .= getindex.(I, 1)
  cols .= getindex.(I, 2)
  return rows, cols
end

function NLPModels.hess_coord!(
  nls::SimpleNLSModel,
  x::AbstractVector,
  vals::AbstractVector;
  obj_weight = 1.0,
)
  @lencheck 2 x
  @lencheck 3 vals
  Hx = hess(nls, x, obj_weight = obj_weight)
  k = 1
  for j = 1:2
    for i = j:2
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function NLPModels.hess_coord!(
  nls::SimpleNLSModel,
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight = 1.0,
)
  @lencheck 2 x
  @lencheck 3 y
  @lencheck 3 vals
  Hx = hess(nls, x, y, obj_weight = obj_weight)
  k = 1
  for j = 1:2
    for i = j:2
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function NLPModels.hprod!(
  nls::SimpleNLSModel,
  x::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x v Hv
  increment!(nls, :neval_hprod)
  Hv .= obj_weight * [T(1) - 200 * x[2]+600 * x[1]^2 -200*x[1]; -200*x[1] T(100)] * v
  return Hv
end

function NLPModels.hprod!(
  nls::SimpleNLSModel,
  x::AbstractVector{T},
  y::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x v Hv
  increment!(nls, :neval_hprod)
  Hv .=
    [
      obj_weight*(T(1) - 200 * x[2] + 600 * x[1]^2)+2*y[2]+2*y[3] -obj_weight*200*x[1]
      -obj_weight*200*x[1] obj_weight*T(100)+2*y[1]+2*y[3]
    ] * v
  return Hv
end

function NLPModels.ghjvprod!(
  nls::SimpleNLSModel,
  x::AbstractVector{T},
  g::AbstractVector{T},
  v::AbstractVector{T},
  gHv::AbstractVector{T},
) where {T}
  @lencheck nls.meta.nvar x g v
  @lencheck nls.meta.ncon gHv
  increment!(nls, :neval_hprod)
  gHv[1] = g[2] * 2v[2]
  gHv[2] = g[1] * 2v[1]
  gHv[3] = g[1] * 2v[1] + g[2] * 2v[2]
  return gHv
end
