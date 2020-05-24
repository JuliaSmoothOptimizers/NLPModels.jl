using NLPModels: increment!

#Problem 5 in the Hock-Schittkowski suite
function hs5_autodiff()

  x0 = [0.0; 0.0]
  f(x) = sin(x[1] + x[2]) + (x[1] - x[2])^2 - 3x[1] / 2 + 5x[2] / 2 + 1
  l = [-1.5; -3.0]
  u = [4.0; 3.0]

  return ADNLPModel(f, x0, l, u, name="hs5_autodiff")
end

mutable struct HS5 <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
end

function HS5()
  meta = NLPModelMeta(2, x0=zeros(2), lvar=[-1.5; -3.0], uvar=[4.0; 3.0], name="HS5_manual")

  return HS5(meta, Counters())
end

function NLPModels.obj(nlp :: HS5, x :: AbstractVector)
  @lencheck 2 x
  increment!(nlp, :neval_obj)
  return sin(x[1] + x[2]) + (x[1] - x[2])^2 - 3x[1] / 2 + 5x[2] / 2 + 1
end

function NLPModels.grad!(nlp :: HS5, x :: AbstractVector{T}, gx :: AbstractVector{T}) where T
  @lencheck 2 x gx
  increment!(nlp, :neval_grad)
  gx .= cos(x[1] + x[2]) * ones(T, 2) + 2 * (x[1] - x[2]) * T[1; -1] + T[-1.5; 2.5]
  return gx
end

function NLPModels.hess_structure!(nlp :: HS5, rows :: AbstractVector{Int}, cols :: AbstractVector{Int})
  @lencheck 3 rows cols
  rows .= [1; 2; 2]
  cols .= [1; 1; 2]
  return rows, cols
end

function NLPModels.hess_coord!(nlp :: HS5, x :: AbstractVector, vals :: AbstractVector; obj_weight=1.0)
  @lencheck 2 x
  @lencheck 3 vals
  increment!(nlp, :neval_hess)
  vals[1] = vals[3] = -sin(x[1] + x[2]) + 2
  vals[2] = -sin(x[1] + x[2]) - 2
  vals .*= obj_weight
  return vals
end

function NLPModels.hprod!(nlp :: HS5, x :: AbstractVector{T}, v :: AbstractVector{T}, Hv :: AbstractVector{T}; obj_weight=one(T)) where T
  @lencheck 2 x v Hv
  increment!(nlp, :neval_hprod)
  Hv .= (- sin(x[1] + x[2]) * (v[1] + v[2]) * ones(T, 2) + 2 * [v[1] - v[2]; v[2] - v[1]]) * obj_weight
  return Hv
end
