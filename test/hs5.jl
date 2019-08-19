using NLPModels: increment!

#Problem 5 in the Hock-Schittkowski suite
mutable struct HS5 <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
end

function HS5()
  meta = NLPModelMeta(2, x0=zeros(2), lvar=[-1.5; -3.0], uvar=[4.0; 3.0], name="hs5")

  return HS5(meta, Counters())
end

function NLPModels.obj(nlp :: HS5, x :: AbstractVector)
  increment!(nlp, :neval_obj)
  return sin(x[1] + x[2]) + (x[1] - x[2])^2 - 1.5 * x[1] + 2.5 * x[2] + 1
end

function NLPModels.grad!(nlp :: HS5, x :: AbstractVector, gx :: AbstractVector)
  increment!(nlp, :neval_grad)
  gx .= cos(x[1] + x[2]) * ones(2) + 2 * (x[1] - x[2]) * [1.0; -1.0] + [-1.5; 2.5]
  return gx
end

function NLPModels.hess(nlp :: HS5, x :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hess)
  return tril(-sin(x[1] + x[2])*ones(2, 2) + [2.0 -2.0; -2.0 2.0]) * obj_weight
end

function NLPModels.hess_structure!(nlp :: HS5, rows :: AbstractVector{Int}, cols :: AbstractVector{Int})
  I = ((i,j) for i = 1:nlp.meta.nvar, j = 1:nlp.meta.nvar if i ≥ j)
  rows .= getindex.(I, 1)
  cols .= getindex.(I, 2)
  return rows, cols
end

function NLPModels.hess_coord!(nlp :: HS5, x :: AbstractVector, rows :: AbstractVector{Int}, cols :: AbstractVector{Int}, vals :: AbstractVector; obj_weight=1.0, y=AbstractVector[])
  H = hess(nlp, x, obj_weight=obj_weight)
  nnzh = length(vals)
  for k = 1:nnzh
    i, j = rows[k], cols[k]
    vals[k] = H[i,j]
  end
  return rows, cols, vals
end

function NLPModels.hess_coord(nlp :: HS5, x :: AbstractVector; obj_weight=1.0, y=Float64[])
  H = hess(nlp, x, obj_weight=obj_weight)
  I = ((i,j,H[i,j]) for i = 1:nlp.meta.nvar, j = 1:nlp.meta.nvar if i ≥ j)
  return (getindex.(I, 1), getindex.(I, 2), getindex.(I, 3))
end

function NLPModels.hprod!(nlp :: HS5, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hprod)
  Hv .= (- sin(x[1] + x[2]) * (v[1] + v[2]) * ones(2) + 2 * [v[1] - v[2]; v[2] - v[1]]) * obj_weight
  return Hv
end
