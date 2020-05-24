using NLPModels: increment!

#Problem 6 in the Hock-Schittkowski suite
function hs6_autodiff()
  x0 = [-1.2; 1.0]
  f(x) = (1 - x[1])^2
  c(x) = [10 * (x[2] - x[1]^2)]
  lcon = [0.0]
  ucon = [0.0]

  return ADNLPModel(f, x0, c, lcon, ucon, name="hs6_autodiff")
end

mutable struct HS6 <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
end

function HS6()
  meta = NLPModelMeta(2, ncon=1, nnzh=1, nnzj=2, x0=[-1.2; 1.0], lcon=[0.0], ucon=[0.0], name="HS6_manual")

  return HS6(meta, Counters())
end

function NLPModels.obj(nlp :: HS6, x :: AbstractVector)
  @lencheck 2 x
  increment!(nlp, :neval_obj)
  return (1 - x[1])^2
end

function NLPModels.grad!(nlp :: HS6, x :: AbstractVector, gx :: AbstractVector)
  @lencheck 2 x gx
  increment!(nlp, :neval_grad)
  gx .= [2 * (x[1] - 1); 0]
  return gx
end

function NLPModels.hess_structure!(nlp :: HS6, rows :: AbstractVector{Int}, cols :: AbstractVector{Int})
  @lencheck 1 rows cols
  rows[1] = 1
  cols[1] = 1
  return rows, cols
end

function NLPModels.hess_coord!(nlp :: HS6, x :: AbstractVector{T}, vals :: AbstractVector{T}; obj_weight=one(T)) where T
  @lencheck 2 x
  @lencheck 1 vals
  increment!(nlp, :neval_hess)
  vals[1] = 2obj_weight
  return vals
end

function NLPModels.hess_coord!(nlp :: HS6, x :: AbstractVector{T}, y :: AbstractVector{T}, vals :: AbstractVector{T}; obj_weight=one(T)) where T
  @lencheck 2 x
  @lencheck 1 y vals
  increment!(nlp, :neval_hess)
  vals[1] = 2obj_weight - 20y[1]
  return vals
end

function NLPModels.hprod!(nlp :: HS6, x :: AbstractVector{T}, y :: AbstractVector{T}, v :: AbstractVector{T}, Hv :: AbstractVector{T}; obj_weight=one(T)) where T
  @lencheck 2 x v Hv
  @lencheck 1 y
  increment!(nlp, :neval_hprod)
  Hv .= [(2obj_weight - 20y[1]) * v[1]; 0]
  return Hv
end

function NLPModels.cons!(nlp :: HS6, x :: AbstractVector, cx :: AbstractVector)
  @lencheck 2 x
  @lencheck 1 cx
  increment!(nlp, :neval_cons)
  cx[1] = 10 * (x[2] - x[1]^2)
  return cx
end

function NLPModels.jac_structure!(nlp :: HS6, rows :: AbstractVector{Int}, cols :: AbstractVector{Int})
  @lencheck 2 rows cols
  rows[1:2] .= [1, 1]
  cols[1:2] .= [1, 2]
  return rows, cols
end

function NLPModels.jac_coord!(nlp :: HS6, x :: AbstractVector, vals :: AbstractVector)
  @lencheck 2 x vals
  increment!(nlp, :neval_jac)
  vals[1] = -20 * x[1]
  vals[2] = 10
  return vals
end

function NLPModels.jprod!(nlp :: HS6, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck 2 x v
  @lencheck 1 Jv
  increment!(nlp, :neval_jprod)
  Jv .= [-20 * x[1] * v[1] + 10 * v[2]]
  return Jv
end

function NLPModels.jtprod!(nlp :: HS6, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck 2 x Jtv
  @lencheck 1 v
  increment!(nlp, :neval_jtprod)
  Jtv .= [-20 * x[1]; 10] * v[1]
  return Jtv
end
