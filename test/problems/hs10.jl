using NLPModels: increment!

#Problem 10 in the Hock-Schittkowski suite
function hs10_autodiff()

  x0 = [-10.0; 10.0]
  f(x) = x[1] - x[2]
  c(x) = [-3 * x[1]^2 + 2 * x[1] * x[2] - x[2]^2 + 1]
  lcon = [0.0]
  ucon = [Inf]

  return ADNLPModel(f, x0, c, lcon, ucon, name="hs10_autodiff")
end

mutable struct HS10 <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
end

function HS10()
  meta = NLPModelMeta(2, ncon=1, x0=[-10.0; 10.0],
                      lcon=[0.0], ucon=[Inf], name="HS10_manual")

  return HS10(meta, Counters())
end

function NLPModels.obj(nlp :: HS10, x :: AbstractVector)
  @lencheck 2 x
  increment!(nlp, :neval_obj)
  return x[1] - x[2]
end

function NLPModels.grad!(nlp :: HS10, x :: AbstractVector{T}, gx :: AbstractVector{T}) where T
  @lencheck 2 x gx
  increment!(nlp, :neval_grad)
  gx .= T[1; -1]
  return gx
end

function NLPModels.hess_structure!(nlp :: HS10, rows :: AbstractVector{Int}, cols :: AbstractVector{Int})
  @lencheck 3 rows cols
  rows[1] = 1; rows[2] = 2; rows[3] = 2
  cols[1] = 1; cols[2] = 1; cols[3] = 2
  return rows, cols
end

function NLPModels.hess_coord!(nlp :: HS10, x :: AbstractVector{T}, vals :: AbstractVector{T}; obj_weight=1.0) where T
  @lencheck 2 x
  @lencheck 3 vals
  increment!(nlp, :neval_hess)
  vals .= zero(T)
  return vals
end

function NLPModels.hess_coord!(nlp :: HS10, x :: AbstractVector{T}, y :: AbstractVector{T}, vals :: AbstractVector{T}; obj_weight=1.0) where T
  @lencheck 2 x
  @lencheck 1 y
  @lencheck 3 vals
  increment!(nlp, :neval_hess)
  vals .= T[-6, 2, -2] * y[1]
  return vals
end

function NLPModels.hprod!(nlp :: HS10, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight=1.0)
  @lencheck 2 x v Hv
  @lencheck 1 y
  increment!(nlp, :neval_hprod)
  Hv .= y[1] * [-6 * v[1] + 2 * v[2]; 2 * v[1] - 2 * v[2]]
  return Hv
end

function NLPModels.cons!(nlp :: HS10, x :: AbstractVector, cx :: AbstractVector)
  @lencheck 2 x
  @lencheck 1 cx
  increment!(nlp, :neval_cons)
  cx .= [-3 * x[1]^2 + 2 * x[1] * x[2] - x[2]^2 + 1]
  return cx
end

function NLPModels.jac_structure!(nlp :: HS10, rows :: AbstractVector{Int}, cols :: AbstractVector{Int})
  @lencheck 2 rows cols
  rows .= [1, 1]
  cols .= [1, 2]
  return rows, cols
end

function NLPModels.jac_coord!(nlp :: HS10, x :: AbstractVector, vals :: AbstractVector)
  @lencheck 2 x vals
  increment!(nlp, :neval_jac)
  vals .= [-6 * x[1] + 2 * x[2], 2 * x[1] - 2 * x[2]]
  return vals
end

function NLPModels.jprod!(nlp :: HS10, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck 2 x v
  @lencheck 1 Jv
  increment!(nlp, :neval_jprod)
  Jv .= [(-6 * x[1] + 2 * x[2]) * v[1] + (2 * x[1] - 2 * x[2]) * v[2]]
  return Jv
end

function NLPModels.jtprod!(nlp :: HS10, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck 2 x Jtv
  @lencheck 1 v
  increment!(nlp, :neval_jtprod)
  Jtv .= [-6 * x[1] + 2 * x[2];  2 * x[1] - 2 * x[2]] * v[1]
  return Jtv
end
