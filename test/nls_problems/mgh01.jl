using NLPModels: increment!

function mgh01_autodiff()

  x0 = [-1.2; 1.0]
  F(x) = [10 * (x[2] - x[1]^2); 1 - x[1]]

  return ADNLSModel(F, x0, 2, name="mgh01_autodiff")
end

mgh01_special() = FeasibilityResidual(MGH01Feas())

mutable struct MGH01 <: AbstractNLSModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  counters :: NLSCounters
end

function MGH01()
  meta = NLPModelMeta(2, x0=[-1.2; 1.0], name="MGH01_manual")
  nls_meta = NLSMeta(2, 2, nnzj=3, nnzh=1)

  return MGH01(meta, nls_meta, NLSCounters())
end

function NLPModels.residual!(nls :: MGH01, x :: AbstractVector, Fx :: AbstractVector)
  increment!(nls, :neval_residual)
  Fx[1:2] .= [10 * (x[2] - x[1]^2); 1 - x[1]]
  return Fx
end

# Jx = [-20x₁  10;  -1  0]
function NLPModels.jac_structure_residual!(nls :: MGH01, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  rows[1:3] .= [1, 1, 2]
  cols[1:3] .= [1, 2, 1]
  return rows, cols
end

function NLPModels.jac_coord_residual!(nls :: MGH01, x :: AbstractVector, vals :: AbstractVector)
  increment!(nls, :neval_jac_residual)
  vals[1:3] .= [-20x[1], 10, -1]
  return vals
end

function NLPModels.jprod_residual!(nls :: MGH01, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nls, :neval_jprod_residual)
  Jv[1:2] .= [-20x[1] * v[1] + 10v[2]; -v[1]]
  return Jv
end

function NLPModels.jtprod_residual!(nls :: MGH01, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nls, :neval_jtprod_residual)
  Jtv[1:2] .= [-20x[1] * v[1] - v[2]; 10v[1]]
  return Jtv
end

function NLPModels.hess_structure_residual!(nls :: MGH01, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  rows[1] = 1
  cols[1] = 1
  return rows, cols
end

function NLPModels.hess_coord_residual!(nls :: MGH01, x :: AbstractVector, v :: AbstractVector, vals :: AbstractVector)
  increment!(nls, :neval_hess_residual)
  vals[1] = -20v[1]
  return vals
end

function NLPModels.jth_hess_residual(nls :: MGH01, x :: AbstractVector, j :: Int)
  increment!(nls, :neval_jhess_residual)
  if j == 1
    return eltype(x)[-20 0; 0 0]
  else
    return zeros(eltype(x), 2, 2)
  end
end

function NLPModels.hprod_residual!(nls :: MGH01, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  increment!(nls, :neval_hprod_residual)
  if i == 1
    Hiv[1:2] .= [-20v[1]; 0]
  else
    Hiv[1:2] .= zero(eltype(x))
  end
  return Hiv
end

mutable struct MGH01Feas <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
end

function MGH01Feas()
  meta = NLPModelMeta(2, x0=[-1.2; 1.0], name="MGH01Feas_manual", ncon=2, lcon=zeros(2), ucon=zeros(2), nnzj=3, nnzh=1)

  return MGH01Feas(meta, Counters())
end

function NLPModels.cons!(nls :: MGH01Feas, x :: AbstractVector, cx :: AbstractVector)
  increment!(nls, :neval_cons)
  cx[1:2] .= [10 * (x[2] - x[1]^2); 1 - x[1]]
  return cx
end

# Jx = [-20x₁  10;  -1  0]
function NLPModels.jac_structure!(nls :: MGH01Feas, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  rows[1:3] .= [1, 1, 2]
  cols[1:3] .= [1, 2, 1]
  return rows, cols
end

function NLPModels.jac_coord!(nls :: MGH01Feas, x :: AbstractVector, vals :: AbstractVector)
  increment!(nls, :neval_jac)
  vals[1:3] .= [-20x[1], 10, -1]
  return vals
end

function NLPModels.jprod!(nls :: MGH01Feas, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nls, :neval_jprod)
  Jv[1:2] .= [-20x[1] * v[1] + 10v[2]; -v[1]]
  return Jv
end

function NLPModels.jtprod!(nls :: MGH01Feas, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nls, :neval_jtprod)
  Jtv[1:2] .= [-20x[1] * v[1] - v[2]; 10v[1]]
  return Jtv
end

function NLPModels.hess_structure!(nls :: MGH01Feas, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  rows[1] = 1
  cols[1] = 1
  return rows, cols
end

function NLPModels.hess_coord!(nls :: MGH01Feas, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector; obj_weight::Float64=1.0)
  increment!(nls, :neval_hess)
  vals[1] = -20y[1]
  return vals
end

function NLPModels.hprod!(nls :: MGH01Feas, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight::Float64=1.0)
  increment!(nls, :neval_hprod)
  Hv[1:2] .= [-20y[1] * v[1]; 0]
  return Hv
end
