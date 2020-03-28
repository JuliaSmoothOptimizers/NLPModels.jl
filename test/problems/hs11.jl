#Problem 11 in the Hock-Schittkowski suite
function hs11_autodiff()

  x0 = [4.9; 0.1]
  f(x) = (x[1] - 5)^2 + x[2]^2 - 25
  c(x) = [-x[1]^2 + x[2]]
  lcon = [-Inf]
  ucon = [0.0]

  return ADNLPModel(f, x0, c=c, lcon=lcon, ucon=ucon, name="hs11_autodiff")

end

mutable struct HS11 <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
end

function HS11()
  meta = NLPModelMeta(2, ncon=1, nnzh=2, nnzj=2, x0=[4.9; 0.1], lcon=[-Inf], ucon=[0.0], name="HS11_manual")

  return HS11(meta, Counters())
end

function NLPModels.obj(nlp :: HS11, x :: AbstractVector)
  increment!(nlp, :neval_obj)
  return (x[1] - 5)^2 + x[2]^2 - 25
end

function NLPModels.grad!(nlp :: HS11, x :: AbstractVector, gx :: AbstractVector)
  increment!(nlp, :neval_grad)
  gx .= [2 * (x[1] - 5); 2 * x[2]]
  return gx
end

function NLPModels.hess(nlp :: HS11, x :: AbstractVector{T}; obj_weight=one(T)) where T
  increment!(nlp, :neval_hess)
  return T[2 0; 0 2] * obj_weight
end

function NLPModels.hess(nlp :: HS11, x :: AbstractVector{T}, y :: AbstractVector{T}; obj_weight=one(T)) where T
  increment!(nlp, :neval_hess)
  return y[1] * T[-2 0; 0 0] + 2obj_weight*I
end

function NLPModels.hess_structure!(nlp :: HS11, rows :: AbstractVector{Int}, cols :: AbstractVector{Int})
  rows[1] = 1; rows[2] = 2
  cols[1] = 1; cols[2] = 2
  return rows, cols
end

function NLPModels.hess_coord!(nlp :: HS11, x :: AbstractVector{T}, vals :: AbstractVector{T}; obj_weight=one(T)) where T
  increment!(nlp, :neval_hess)
  vals .= 2obj_weight
  return vals
end

function NLPModels.hess_coord!(nlp :: HS11, x :: AbstractVector{T}, y :: AbstractVector{T}, vals :: AbstractVector{T}; obj_weight=one(T)) where T
  increment!(nlp, :neval_hess)
  vals .= 2obj_weight
  vals[1] -= 2y[1]
  return vals
end

function NLPModels.hprod!(nlp :: HS11, x :: AbstractVector{T}, y :: AbstractVector{T}, v :: AbstractVector{T}, Hv :: AbstractVector{T}; obj_weight=one(T)) where T
  increment!(nlp, :neval_hprod)
  Hv .= 2obj_weight * v
  Hv[1] -= 2y[1] * v[1]
  return Hv
end

function NLPModels.cons!(nlp :: HS11, x :: AbstractVector, cx :: AbstractVector)
  increment!(nlp, :neval_cons)
  cx .= [-x[1]^2 + x[2]]
  return cx
end

function NLPModels.jac(nlp :: HS11, x :: AbstractVector)
  increment!(nlp, :neval_jac)
  return [-2 * x[1]  1]
end

function NLPModels.jac_structure!(nlp :: HS11, rows :: AbstractVector{Int}, cols :: AbstractVector{Int})
  rows[1:2] .= [1, 1]
  cols[1:2] .= [1, 2]
  return rows, cols
end

function NLPModels.jac_coord!(nlp :: HS11, x :: AbstractVector, vals :: AbstractVector)
  increment!(nlp, :neval_jac)
  vals .= [-2 * x[1], 1]
  return vals
end

function NLPModels.jprod!(nlp :: HS11, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jprod)
  Jv .= [-2 * x[1] * v[1] + v[2]]
  return Jv
end

function NLPModels.jtprod!(nlp :: HS11, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod)
  Jtv .= [-2 * x[1]; 1] * v[1]
  return Jtv
end
