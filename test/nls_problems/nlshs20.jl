using NLPModels: increment!

function nlshs20_autodiff()

  x0 = [-2.0; 1.0]
  F(x) = [1 - x[1]; 10 * (x[2] - x[1]^2)]
  lvar = [-0.5; -Inf]
  uvar = [0.5; Inf]
  c(x) = [x[1] + x[2]^2; x[1]^2 + x[2]; x[1]^2 + x[2]^2 - 1]
  lcon = zeros(3)
  ucon = fill(Inf, 3)

  return ADNLSModel(F, x0, 2, lvar, uvar, c, lcon, ucon, name="nlshs20_autodiff")
end

mutable struct NLSHS20 <: AbstractNLSModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  counters :: NLSCounters
end

function NLSHS20()
  meta = NLPModelMeta(2, x0=[-2.0; 1.0], name="NLSHS20_manual", lvar=[-0.5; -Inf], uvar=[0.5; Inf], ncon=3, lcon=zeros(3), ucon=fill(Inf, 3), nnzj=6)
  nls_meta = NLSMeta(2, 2, nnzj=3, nnzh=1)

  return NLSHS20(meta, nls_meta, NLSCounters())
end

function NLPModels.residual!(nls :: NLSHS20, x :: AbstractVector, Fx :: AbstractVector)
  @lencheck 2 x Fx
  increment!(nls, :neval_residual)
  Fx .= [1 - x[1]; 10 * (x[2] - x[1]^2)]
  return Fx
end

# Jx = [-1  0; -20x₁  10]
function NLPModels.jac_structure_residual!(nls :: NLSHS20, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck 3 rows cols
  rows .= [1, 2, 2]
  cols .= [1, 1, 2]
  return rows, cols
end

function NLPModels.jac_coord_residual!(nls :: NLSHS20, x :: AbstractVector, vals :: AbstractVector)
  @lencheck 2 x
  @lencheck 3 vals
  increment!(nls, :neval_jac_residual)
  vals .= [-1, -20x[1], 10]
  return vals
end

function NLPModels.jprod_residual!(nls :: NLSHS20, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck 2 x v Jv
  increment!(nls, :neval_jprod_residual)
  Jv .= [-v[1]; - 20 * x[1] * v[1] + 10 * v[2]]
  return Jv
end

function NLPModels.jtprod_residual!(nls :: NLSHS20, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck 2 x v Jtv
  increment!(nls, :neval_jtprod_residual)
  Jtv .= [-v[1] - 20 * x[1] * v[2]; 10 * v[2]]
  return Jtv
end

function NLPModels.hess_structure_residual!(nls :: NLSHS20, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck 1 rows cols
  rows[1] = 1
  cols[1] = 1
  return rows, cols
end

function NLPModels.hess_coord_residual!(nls :: NLSHS20, x :: AbstractVector, v :: AbstractVector, vals :: AbstractVector)
  @lencheck 2 x v
  @lencheck 1 vals
  increment!(nls, :neval_hess_residual)
  vals[1] = -20v[2]
  return vals
end

function NLPModels.hprod_residual!(nls :: NLSHS20, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  @lencheck 2 x v Hiv
  increment!(nls, :neval_hprod_residual)
  if i == 2
    Hiv .= [-20v[1]; 0]
  else
    Hiv .= zero(eltype(x))
  end
  return Hiv
end

function NLPModels.cons!(nls :: NLSHS20, x :: AbstractVector, cx :: AbstractVector)
  @lencheck 2 x
  @lencheck 3 cx
  increment!(nls, :neval_cons)
  cx .= [x[1] + x[2]^2; x[1]^2 + x[2]; x[1]^2 + x[2]^2 - 1]
  return cx
end

function NLPModels.jac_structure!(nls :: NLSHS20, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck 6 rows cols
  rows .= [1, 1, 2, 2, 3, 3]
  cols .= [1, 2, 1, 2, 1, 2]
  return rows, cols
end

function NLPModels.jac_coord!(nls :: NLSHS20, x :: AbstractVector, vals :: AbstractVector)
  @lencheck 2 x
  @lencheck 6 vals
  increment!(nls, :neval_jac)
  vals .= [1, 2x[2], 2x[1], 1, 2x[1], 2x[2]]
  return vals
end

function NLPModels.jprod!(nls :: NLSHS20, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck 2 x v
  @lencheck 3 Jv
  increment!(nls, :neval_jprod)
  Jv .= [v[1] + 2x[2] * v[2]; 2x[1] * v[1] + v[2]; 2x[1] * v[1] + 2x[2] * v[2]]
  return Jv
end

function NLPModels.jtprod!(nls :: NLSHS20, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck 2 x Jtv
  @lencheck 3 v
  increment!(nls, :neval_jtprod)
  Jtv .= [v[1] + 2x[1] * (v[2] + v[3]); v[2] + 2x[2] * (v[1] + v[3])]
  return Jtv
end
