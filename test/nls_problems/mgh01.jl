using NLPModels: increment!

function mgh01_autodiff()

  x0 = [-1.2; 1.0]
  F(x) = [1 - x[1]; 10 * (x[2] - x[1]^2)]

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
  @lencheck 2 x Fx
  increment!(nls, :neval_residual)
  Fx .= [1 - x[1]; 10 * (x[2] - x[1]^2)]
  return Fx
end

# Jx = [-1  0; -20x₁  10]
function NLPModels.jac_structure_residual!(nls :: MGH01, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck 3 rows cols
  rows .= [1, 2, 2]
  cols .= [1, 1, 2]
  return rows, cols
end

function NLPModels.jac_coord_residual!(nls :: MGH01, x :: AbstractVector, vals :: AbstractVector)
  @lencheck 2 x
  @lencheck 3 vals
  increment!(nls, :neval_jac_residual)
  vals .= [-1, -20x[1], 10]
  return vals
end

function NLPModels.jprod_residual!(nls :: MGH01, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck 2 x v Jv
  increment!(nls, :neval_jprod_residual)
  Jv .= [-v[1]; - 20 * x[1] * v[1] + 10 * v[2]]
  return Jv
end

function NLPModels.jtprod_residual!(nls :: MGH01, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck 2 x v Jtv
  increment!(nls, :neval_jtprod_residual)
  Jtv .= [-v[1] - 20 * x[1] * v[2]; 10 * v[2]]
  return Jtv
end

function NLPModels.hess_structure_residual!(nls :: MGH01, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck 1 rows cols
  rows[1] = 1
  cols[1] = 1
  return rows, cols
end

function NLPModels.hess_coord_residual!(nls :: MGH01, x :: AbstractVector, v :: AbstractVector, vals :: AbstractVector)
  @lencheck 2 x v
  @lencheck 1 vals
  increment!(nls, :neval_hess_residual)
  vals[1] = -20v[2]
  return vals
end

function NLPModels.hprod_residual!(nls :: MGH01, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  @lencheck 2 x v Hiv
  increment!(nls, :neval_hprod_residual)
  if i == 2
    Hiv .= [-20v[1]; 0]
  else
    Hiv .= zero(eltype(x))
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
  @lencheck 2 x cx
  increment!(nls, :neval_cons)
  cx .= [1 - x[1]; 10 * (x[2] - x[1]^2)]
  return cx
end

# Jx = [-1  0; -20x₁  10]
function NLPModels.jac_structure!(nls :: MGH01Feas, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck 3 rows cols
  rows .= [1, 2, 2]
  cols .= [1, 1, 2]
  return rows, cols
end

function NLPModels.jac_coord!(nls :: MGH01Feas, x :: AbstractVector, vals :: AbstractVector)
  @lencheck 2 x
  @lencheck 3 vals
  increment!(nls, :neval_jac)
  vals .= [-1, -20x[1], 10]
  return vals
end

function NLPModels.jprod!(nls :: MGH01Feas, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck 2 x v Jv
  increment!(nls, :neval_jprod)
  Jv .= [-v[1]; - 20 * x[1] * v[1] + 10 * v[2]]
  return Jv
end

function NLPModels.jtprod!(nls :: MGH01Feas, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck 2 x v Jtv
  increment!(nls, :neval_jtprod)
  Jtv .= [-v[1] - 20 * x[1] * v[2]; 10 * v[2]]
  return Jtv
end

function NLPModels.hess_structure!(nls :: MGH01Feas, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck 1 rows cols
  rows[1] = 1
  cols[1] = 1
  return rows, cols
end

function NLPModels.hess_coord!(nls :: MGH01Feas, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector; obj_weight::Float64=1.0)
  @lencheck 2 x y
  @lencheck 1 vals
  increment!(nls, :neval_hess)
  vals[1] = -20y[2]
  return vals
end

function NLPModels.hprod!(nls :: MGH01Feas, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight::Float64=1.0)
  @lencheck 2 x y v Hv
  increment!(nls, :neval_hprod)
  Hv .= [-20y[2] * v[1]; 0]
  return Hv
end

function NLPModels.ghjvprod!(nls :: MGH01Feas, x :: AbstractVector{T}, g :: AbstractVector{T}, v :: AbstractVector{T}, gHv :: AbstractVector{T}) where T 
 @lencheck nls.meta.nvar x g v
 @lencheck nls.meta.ncon gHv
 increment!(nls, :neval_hprod)
 gHv .= [T(0); - g[1] * 20v[1]]
 return gHv
end
