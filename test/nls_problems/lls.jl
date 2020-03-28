using NLPModels: increment!

function lls_autodiff()

  x0 = [0.0; 0.0]
  F(x) = [x[1] - x[2]; x[1] + x[2] - 2; x[2] - 2]
  c(x) = [x[1] + x[2]]
  lcon = [0.0]
  ucon = [Inf]

  return ADNLSModel(F, x0, 3, c=c, lcon=lcon, ucon=ucon, name="lls_autodiff")
end

function lls_special()
  return LLSModel([1.0 -1; 1 1; 0 1], [0.0; 2; 2], x0=zeros(2), C=[1.0 1], lcon=[0.0], ucon=[Inf], name="lls_LLSModel")
end


mutable struct LLS <: AbstractNLSModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  counters :: NLSCounters
end

function LLS()
  meta = NLPModelMeta(2, x0=zeros(2), name="LLS_manual", ncon=1, lcon=[0.0], ucon=[Inf], nnzj=2)
  nls_meta = NLSMeta(3, 2, nnzj=5, nnzh=0)

  return LLS(meta, nls_meta, NLSCounters())
end

function NLPModels.residual!(nls :: LLS, x :: AbstractVector, Fx :: AbstractVector)
  increment!(nls, :neval_residual)
  Fx[1:3] .= [x[1] - x[2]; x[1] + x[2] - 2; x[2] - 2]
  return Fx
end

function NLPModels.jac_structure_residual!(nls :: LLS, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  rows[1:5] .= [1, 1, 2, 2, 3]
  cols[1:5] .= [1, 2, 1, 2, 2]
  return rows, cols
end

function NLPModels.jac_coord_residual!(nls :: LLS, x :: AbstractVector, vals :: AbstractVector)
  increment!(nls, :neval_jac_residual)
  T = eltype(x)
  vals[1:5] .= T[1, -1, 1, 1, 1]
  return vals
end

function NLPModels.jprod_residual!(nls :: LLS, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nls, :neval_jprod_residual)
  Jv[1:3] .= [v[1] - v[2]; v[1] + v[2]; v[2]]
  return Jv
end

function NLPModels.jtprod_residual!(nls :: LLS, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nls, :neval_jtprod_residual)
  Jtv[1:2] .= [v[1] + v[2]; -v[1] + v[2] + v[3]]
  return Jtv
end

function NLPModels.hess_structure_residual!(nls :: LLS, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  return rows, cols
end

function NLPModels.hess_coord_residual!(nls :: LLS, x :: AbstractVector, v :: AbstractVector, vals :: AbstractVector)
  increment!(nls, :neval_hess_residual)
  return vals
end

function NLPModels.jth_hess_residual(nls :: LLS, x :: AbstractVector, j :: Int)
  increment!(nls, :neval_jhess_residual)
  return zeros(eltype(x), 2, 2)
end

function NLPModels.hprod_residual!(nls :: LLS, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  increment!(nls, :neval_hprod_residual)
  Hiv[1:2] .= zero(eltype(x))
  return Hiv
end

function NLPModels.cons!(nls :: LLS, x :: AbstractVector, cx :: AbstractVector)
  increment!(nls, :neval_cons)
  cx[1] = x[1] + x[2]
  return cx
end

function NLPModels.jac_structure!(nls :: LLS, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  rows[1:2] .= [1, 1]
  cols[1:2] .= [1, 2]
  return rows, cols
end

function NLPModels.jac_coord!(nls :: LLS, x :: AbstractVector, vals :: AbstractVector)
  increment!(nls, :neval_jac)
  T = eltype(x)
  vals[1:2] .= T[1, 1]
  return vals
end

function NLPModels.jprod!(nls :: LLS, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nls, :neval_jprod)
  Jv[1] = v[1] + v[2]
  return Jv
end

function NLPModels.jtprod!(nls :: LLS, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nls, :neval_jtprod)
  Jtv[1:2] .= v
  return Jtv
end
