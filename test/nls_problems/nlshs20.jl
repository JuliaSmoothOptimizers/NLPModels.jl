using NLPModels: increment!

function nlshs20_autodiff()

  x0 = [-2.0; 1.0]
  F(x) = [10 * (x[2] - x[1]^2); 1 - x[1]]
  lvar = [-0.5; -Inf]
  uvar = [0.5; Inf]
  c(x) = [x[1] + x[2]^2; x[1]^2 + x[2]; x[1]^2 + x[2]^2 - 1]
  lcon = zeros(3)
  ucon = fill(Inf, 3)

  return ADNLSModel(F, x0, 2, lvar=lvar, uvar=uvar, c=c, lcon=lcon, ucon=ucon)
end

mutable struct NLSHS20 <: AbstractNLSModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  counters :: NLSCounters
end

function NLSHS20()
  meta = NLPModelMeta(2, x0=[-2.0; 1.0], name="NLSHS20", lvar=[-0.5; -Inf], uvar=[0.5; Inf], ncon=3, lcon=zeros(3), ucon=fill(Inf, 3), nnzj=6)
  nls_meta = NLSMeta(2, 2, nnzj=3, nnzh=1)

  return NLSHS20(meta, nls_meta, NLSCounters())
end

function NLPModels.residual!(nls :: NLSHS20, x :: AbstractVector, Fx :: AbstractVector)
  increment!(nls, :neval_residual)
  Fx[1:2] .= [10 * (x[2] - x[1]^2); 1 - x[1]]
  return Fx
end

# Jx = [-20x₁  10;  -1  0]
function NLPModels.jac_structure_residual!(nls :: NLSHS20, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  rows[1:3] .= [1, 1, 2]
  cols[1:3] .= [1, 2, 1]
  return rows, cols
end

function NLPModels.jac_coord_residual!(nls :: NLSHS20, x :: AbstractVector, vals :: AbstractVector)
  increment!(nls, :neval_jac_residual)
  vals[1:3] .= [-20x[1], 10, -1]
  return vals
end

function NLPModels.jprod_residual!(nls :: NLSHS20, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nls, :neval_jprod_residual)
  Jv[1:2] .= [-20x[1] * v[1] + 10v[2]; -v[1]]
  return Jv
end

function NLPModels.jtprod_residual!(nls :: NLSHS20, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nls, :neval_jtprod_residual)
  Jtv[1:2] .= [-20x[1] * v[1] - v[2]; 10v[1]]
  return Jtv
end

function NLPModels.hess_structure_residual!(nls :: NLSHS20, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  rows[1] = 1
  cols[1] = 1
  return rows, cols
end

function NLPModels.hess_coord_residual!(nls :: NLSHS20, x :: AbstractVector, v :: AbstractVector, vals :: AbstractVector)
  increment!(nls, :neval_hess_residual)
  vals[1] = -20v[1]
  return vals
end

function NLPModels.jth_hess_residual(nls :: NLSHS20, x :: AbstractVector, j :: Int)
  increment!(nls, :neval_jhess_residual)
  if j == 1
    return eltype(x)[-20 0; 0 0]
  else
    return zeros(eltype(x), 2, 2)
  end
end

function NLPModels.hprod_residual!(nls :: NLSHS20, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  increment!(nls, :neval_hprod_residual)
  if i == 1
    Hiv[1:2] .= [-20v[1]; 0]
  else
    Hiv[1:2] .= zero(eltype(x))
  end
  return Hiv
end

function NLPModels.cons!(nls :: NLSHS20, x :: AbstractVector, cx :: AbstractVector)
  increment!(nls, :neval_cons)
  cx[1:3] .= [x[1] + x[2]^2; x[1]^2 + x[2]; x[1]^2 + x[2]^2 - 1]
  return cx
end

function NLPModels.jac_structure!(nls :: NLSHS20, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  rows[1:6] .= [1, 1, 2, 2, 3, 3]
  cols[1:6] .= [1, 2, 1, 2, 1, 2]
  return rows, cols
end

function NLPModels.jac_coord!(nls :: NLSHS20, x :: AbstractVector, vals :: AbstractVector)
  increment!(nls, :neval_jac)
  vals[1:6] .= [1, 2x[2], 2x[1], 1, 2x[1], 2x[2]]
  return vals
end

function NLPModels.jprod!(nls :: NLSHS20, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nls, :neval_jprod)
  Jv[1:3] .= [v[1] + 2x[2] * v[2]; 2x[1] * v[1] + v[2]; 2x[1] * v[1] + 2x[2] * v[2]]
  return Jv
end

function NLPModels.jtprod!(nls :: NLSHS20, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nls, :neval_jtprod)
  Jtv[1:2] .= [v[1] + 2x[1] * (v[2] + v[3]); v[2] + 2x[2] * (v[1] + v[3])]
  return Jtv
end
