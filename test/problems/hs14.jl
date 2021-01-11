#Problem 14 in the Hock-Schittkowski suite
function hs14_autodiff()

  x0 = [2.0; 2.0]
  f(x) = (x[1] - 2)^2 + (x[2] - 1)^2
  c(x) = [x[1] - 2 * x[2] + 1; -x[1]^2/4 - x[2]^2 + 1]
  lcon = [0.0; 0.0]
  ucon = [0.0; Inf]

  return ADNLPModel(f, x0, c, lcon, ucon, name="hs14_autodiff")
end

mutable struct HS14 <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
end

function HS14()
  meta = NLPModelMeta(2, nnzh=2, ncon=2, x0=[2.0; 2.0], lcon=[0.0; 0.0], ucon=[0.0; Inf], name="HS14_manual")

  return HS14(meta, Counters())
end

function NLPModels.obj(nlp :: HS14, x :: AbstractVector)
  @lencheck 2 x
  increment!(nlp, :neval_obj)
  return (x[1] - 2)^2 + (x[2] - 1)^2
end

function NLPModels.grad!(nlp :: HS14, x :: AbstractVector, gx :: AbstractVector)
  @lencheck 2 x gx
  increment!(nlp, :neval_grad)
  gx .= [2 * (x[1] - 2); 2 * (x[2] - 1)]
  return gx
end

function NLPModels.hess_structure!(nlp :: HS14, rows :: AbstractVector{Int}, cols :: AbstractVector{Int})
  @lencheck 2 rows cols
  rows[1] = 1; rows[2] = 2
  cols[1] = 1; cols[2] = 2
  return rows, cols
end

function NLPModels.hess_coord!(nlp :: HS14, x :: AbstractVector{T}, vals :: AbstractVector{T}; obj_weight=one(T)) where T
  @lencheck 2 x vals
  increment!(nlp, :neval_hess)
  vals .= 2obj_weight
  return vals
end

function NLPModels.hess_coord!(nlp :: HS14, x :: AbstractVector{T}, y :: AbstractVector{T}, vals :: AbstractVector{T}; obj_weight=one(T)) where T
  @lencheck 2 x y vals
  increment!(nlp, :neval_hess)
  vals .= 2obj_weight
  vals[1] -= y[2] / 2
  vals[2] -= 2y[2]
  return vals
end

function NLPModels.hprod!(nlp :: HS14, x :: AbstractVector{T}, y :: AbstractVector{T}, v :: AbstractVector{T}, Hv :: AbstractVector{T}; obj_weight=one(T)) where T
  @lencheck 2 x y v Hv
  increment!(nlp, :neval_hprod)
  Hv .= 2obj_weight * v
  Hv[1] -= y[2] * v[1] / 2
  Hv[2] -= 2y[2] * v[2]
  return Hv
end

function NLPModels.cons!(nlp :: HS14, x :: AbstractVector, cx :: AbstractVector)
  @lencheck 2 x cx
  increment!(nlp, :neval_cons)
  cx .= [x[1] - 2 * x[2] + 1; -x[1]^2/4 - x[2]^2 + 1]
  return cx
end

function NLPModels.jac_structure!(nlp :: HS14, rows :: AbstractVector{Int}, cols :: AbstractVector{Int})
  @lencheck 4 rows cols
  rows .= [1, 2, 1, 2]
  cols .= [1, 1, 2, 2]
  return rows, cols
end

function NLPModels.jac_coord!(nlp :: HS14, x :: AbstractVector, vals :: AbstractVector)
  @lencheck 2 x
  @lencheck 4 vals
  increment!(nlp, :neval_jac)
  vals .= [1, -x[1] / 2, -2, -2 * x[2]]
  return vals
end

function NLPModels.jprod!(nlp :: HS14, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck 2 x v Jv
  increment!(nlp, :neval_jprod)
  Jv .= [v[1] - 2 * v[2]; -x[1] * v[1] / 2 - 2 * x[2] * v[2]]
  return Jv
end

function NLPModels.jtprod!(nlp :: HS14, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck 2 x v Jtv
  increment!(nlp, :neval_jtprod)
  Jtv .= [v[1] - x[1] * v[2] / 2; -2 * v[1] - 2 * x[2] * v[2]]
  return Jtv
end
