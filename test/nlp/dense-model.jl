"""
    ManualDenseNLPModel <: AbstractDenseNLPModel

Concrete dense NLP model for demonstration and testing.
This model stores Jacobian and Hessian as dense matrices.

Example problem:
    min   x₁² + x₂²
    s.to  x₁ + x₂ = 1
          x₁² + x₂² ≤ 2

x₀ = [0.5, 0.5]
"""
mutable struct ManualDenseNLPModel{T, S} <: AbstractDenseNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
end

function ManualDenseNLPModel(::Type{T}) where {T}
  meta = NLPModelMeta(
    2,                      # nvar
    ncon = 2,              # number of constraints
    nnzj = 4,              # 2x2 dense Jacobian = 4 entries
    nnzh = 3,              # lower triangle of 2x2 Hessian = 3 entries
    x0 = T[0.5, 0.5],
    lcon = T[-Inf, -Inf],
    ucon = T[0, 0],        # x₁ + x₂ = 0 (shift by -1), x₁² + x₂² ≤ 2
    name = "Manual Dense NLP Model",
    lin = [1],             # first constraint is linear
    lin_nnzj = 2,
    nln_nnzj = 2,
  )

  return ManualDenseNLPModel(meta, Counters())
end

ManualDenseNLPModel() = ManualDenseNLPModel(Float64)

# Objective: f(x) = x₁² + x₂²
function NLPModels.obj(nlp::ManualDenseNLPModel, x::AbstractVector)
  @lencheck 2 x
  increment!(nlp, :neval_obj)
  return x[1]^2 + x[2]^2
end

function NLPModels.grad!(nlp::ManualDenseNLPModel, x::AbstractVector, gx::AbstractVector)
  @lencheck 2 x gx
  increment!(nlp, :neval_grad)
  gx[1] = 2 * x[1]
  gx[2] = 2 * x[2]
  return gx
end

# Constraints: c₁(x) = x₁ + x₂ - 1, c₂(x) = x₁² + x₂² - 2
function NLPModels.cons_lin!(nlp::ManualDenseNLPModel, x::AbstractVector, cx::AbstractVector)
  @lencheck 2 x
  @lencheck 1 cx
  increment!(nlp, :neval_cons_lin)
  cx[1] = x[1] + x[2] - 1
  return cx
end

function NLPModels.cons_nln!(nlp::ManualDenseNLPModel, x::AbstractVector, cx::AbstractVector)
  @lencheck 2 x
  @lencheck 1 cx
  increment!(nlp, :neval_cons_nln)
  cx[1] = x[1]^2 + x[2]^2 - 2
  return cx
end

# Jacobian structure for dense model (all entries)
function NLPModels.jac_structure!(
  nlp::ManualDenseNLPModel,
  rows::AbstractVector{Int},
  cols::AbstractVector{Int},
)
  @lencheck 4 rows cols
  # For 2 constraints × 2 variables = 4 entries
  # Row-major order: (1,1), (1,2), (2,1), (2,2)
  rows .= [1, 1, 2, 2]
  cols .= [1, 2, 1, 2]
  return rows, cols
end

function NLPModels.jac_lin_structure!(
  nlp::ManualDenseNLPModel,
  rows::AbstractVector{Int},
  cols::AbstractVector{Int},
)
  @lencheck 2 rows cols
  # Linear constraint: c₁ = x₁ + x₂ - 1
  # ∇c₁ = [1, 1]
  rows .= [1, 1]
  cols .= [1, 2]
  return rows, cols
end

function NLPModels.jac_nln_structure!(
  nlp::ManualDenseNLPModel,
  rows::AbstractVector{Int},
  cols::AbstractVector{Int},
)
  @lencheck 2 rows cols
  # Nonlinear constraint: c₂ = x₁² + x₂² - 2
  # ∇c₂ = [2x₁, 2x₂]
  rows .= [1, 1]
  cols .= [1, 2]
  return rows, cols
end

# Jacobian coordinate values
function NLPModels.jac_coord!(
  nlp::ManualDenseNLPModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  @lencheck 2 x
  @lencheck 4 vals
  increment!(nlp, :neval_jac)
  # c₁: [1, 1]
  vals[1] = 1.0
  vals[2] = 1.0
  # c₂: [2x₁, 2x₂]
  vals[3] = 2 * x[1]
  vals[4] = 2 * x[2]
  return vals
end

function NLPModels.jac_lin_coord!(nlp::ManualDenseNLPModel, vals::AbstractVector)
  @lencheck 2 vals
  increment!(nlp, :neval_jac_lin)
  vals .= 1.0
  return vals
end

function NLPModels.jac_nln_coord!(
  nlp::ManualDenseNLPModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  @lencheck 2 x vals
  increment!(nlp, :neval_jac_nln)
  vals[1] = 2 * x[1]
  vals[2] = 2 * x[2]
  return vals
end

# Jacobian-vector products
function NLPModels.jprod_lin!(
  nlp::ManualDenseNLPModel,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck 2 v
  @lencheck 1 Jv
  increment!(nlp, :neval_jprod_lin)
  Jv[1] = v[1] + v[2]
  return Jv
end

function NLPModels.jprod_nln!(
  nlp::ManualDenseNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck 2 x v
  @lencheck 1 Jv
  increment!(nlp, :neval_jprod_nln)
  Jv[1] = 2 * x[1] * v[1] + 2 * x[2] * v[2]
  return Jv
end

function NLPModels.jtprod_lin!(
  nlp::ManualDenseNLPModel,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck 1 v
  @lencheck 2 Jtv
  increment!(nlp, :neval_jtprod_lin)
  Jtv[1] = v[1]
  Jtv[2] = v[1]
  return Jtv
end

function NLPModels.jtprod_nln!(
  nlp::ManualDenseNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck 2 x Jtv
  @lencheck 1 v
  increment!(nlp, :neval_jtprod_nln)
  Jtv[1] = 2 * x[1] * v[1]
  Jtv[2] = 2 * x[2] * v[1]
  return Jtv
end

# Hessian structure (lower triangle)
function NLPModels.hess_structure!(
  nlp::ManualDenseNLPModel,
  rows::AbstractVector{Int},
  cols::AbstractVector{Int},
)
  @lencheck 3 rows cols
  # Lower triangle of 2×2: (1,1), (2,1), (2,2)
  rows .= [1, 2, 2]
  cols .= [1, 1, 2]
  return rows, cols
end

# Lagrangian Hessian: ∇²L = obj_weight * ∇²f + y₁ * ∇²c₁ + y₂ * ∇²c₂
# ∇²f = [2, 0; 0, 2]
# ∇²c₁ = [0, 0; 0, 0] (linear)
# ∇²c₂ = [2, 0; 0, 2]
function NLPModels.hess_coord!(
  nlp::ManualDenseNLPModel,
  x::AbstractVector{T},
  y::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x y
  @lencheck 3 vals
  increment!(nlp, :neval_hess)
  # Lower triangle: (1,1), (2,1), (2,2)
  vals[1] = 2 * obj_weight + 2 * y[2]  # (1,1)
  vals[2] = 0                           # (2,1)
  vals[3] = 2 * obj_weight + 2 * y[2]  # (2,2)
  return vals
end

function NLPModels.hprod!(
  nlp::ManualDenseNLPModel,
  x::AbstractVector{T},
  y::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x y v Hv
  increment!(nlp, :neval_hprod)
  # H = diag([2*obj_weight + 2*y[2], 2*obj_weight + 2*y[2]])
  d = 2 * obj_weight + 2 * y[2]
  Hv[1] = d * v[1]
  Hv[2] = d * v[2]
  return Hv
end
