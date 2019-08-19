#Problem 14 in the Hock-Schittkowski suite
mutable struct HS14 <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
end

function HS14()
  meta = NLPModelMeta(2, nnzh=2, ncon=2, x0=[2.0; 2.0], lcon=[0.0; 0.0], ucon=[0.0; Inf], name="HS14")

  return HS14(meta, Counters())
end

function NLPModels.obj(nlp :: HS14, x :: AbstractVector)
  increment!(nlp, :neval_obj)
  return (x[1] - 2)^2 + (x[2] - 1)^2
end

function NLPModels.grad!(nlp :: HS14, x :: AbstractVector, gx :: AbstractVector)
  increment!(nlp, :neval_grad)
  gx .= [2 * (x[1] - 2); 2 * (x[2] - 1)]
  return gx
end

function NLPModels.hess(nlp :: HS14, x :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hess)
  if length(y) > 0
    return y[2] * [-0.5 0.0; 0.0 -2.0] + 2obj_weight * I
  else
    return [2.0 0; 0 2] * obj_weight
  end
end

function NLPModels.hess_structure!(nlp :: HS14, rows :: AbstractVector{Int}, cols :: AbstractVector{Int})
  rows[1] = 1; rows[2] = 2
  cols[1] = 1; cols[2] = 2
  return rows, cols
end

function NLPModels.hess_coord!(nlp :: HS14, x :: AbstractVector, rows :: AbstractVector{Int}, cols :: AbstractVector{Int}, vals :: AbstractVector; obj_weight=1.0, y=AbstractVector[])
  increment!(nlp, :neval_hess)
  vals .= 2obj_weight
  if length(y) > 0
    vals[1] -= 0.5y[2]
    vals[2] -= 2.0y[2]
  end
  return rows, cols, vals
end

function NLPModels.hess_coord(nlp :: HS14, x :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hess)
  w = length(y) == 0 ? 0.0 : y[2]
  return ([1, 2], [1, 2], [-0.5w + 2obj_weight, -2w + 2obj_weight])
end

function NLPModels.hprod!(nlp :: HS14, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hprod)
  nlp.counters.neval_hess -= 1
  Hx = Symmetric(hess(nlp, x, obj_weight=obj_weight, y=y), :L)
  Hv .= Hx * v
  return Hv
end

function NLPModels.cons!(nlp :: HS14, x :: AbstractVector, cx :: AbstractVector)
  increment!(nlp, :neval_cons)
  cx .= [x[1] - 2 * x[2] + 1; -x[1]^2/4 - x[2]^2 + 1]
  return cx
end

function NLPModels.jac(nlp :: HS14, x :: AbstractVector)
  increment!(nlp, :neval_jac)
  return [1.0 -2.0; -x[1] / 2  -2 * x[2]]
end

function NLPModels.jac_structure!(nlp :: HS14, rows :: AbstractVector{Int}, cols :: AbstractVector{Int})
  rows[1:4] .= [1, 2, 1, 2]
  cols[1:4] .= [1, 1, 2, 2]
  return rows, cols
end

function NLPModels.jac_coord!(nlp :: HS14, x :: AbstractVector, rows :: AbstractVector{Int}, cols :: AbstractVector{Int}, vals :: AbstractVector)
  increment!(nlp, :neval_jac)
  vals .= [1.0, -x[1] / 2, -2.0, -2 * x[2]]
  return rows, cols, vals
end

function NLPModels.jac_coord(nlp :: HS14, x :: AbstractVector)
  increment!(nlp, :neval_jac)
  return ([1, 2, 1, 2], [1, 1, 2, 2], [1.0, -x[1] / 2, -2.0, -2 * x[2]])
end

function NLPModels.jprod!(nlp :: HS14, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jprod)
  Jv .= [v[1] - 2 * v[2]; -x[1] * v[1] / 2 - 2 * x[2] * v[2]]
  return Jv
end

function NLPModels.jtprod!(nlp :: HS14, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod)
  Jtv .= [v[1] - x[1] * v[2] / 2; -2 * v[1] - 2 * x[2] * v[2]]
  return Jtv
end
