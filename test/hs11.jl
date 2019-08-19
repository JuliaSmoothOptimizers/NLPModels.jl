#Problem 11 in the Hock-Schittkowski suite
mutable struct HS11 <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
end

function HS11()
  meta = NLPModelMeta(2, ncon=1, nnzh=2, nnzj=2, x0=[4.9; 0.1], lcon=[-Inf], ucon=[0.0], name="HS11")

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

function NLPModels.hess(nlp :: HS11, x :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hess)
  if length(y) > 0
    return y[1] * [-2.0 0; 0 0] + 2obj_weight*I
  else
    return [2.0 0; 0 2] * obj_weight
  end
end

function NLPModels.hess_structure!(nlp :: HS11, rows :: AbstractVector{Int}, cols :: AbstractVector{Int})
  rows[1] = 1; rows[2] = 2
  cols[1] = 1; cols[2] = 2
  return rows, cols
end

function NLPModels.hess_coord!(nlp :: HS11, x :: AbstractVector, rows :: AbstractVector{Int}, cols :: AbstractVector{Int}, vals :: AbstractVector; obj_weight=1.0, y=AbstractVector[])
  increment!(nlp, :neval_hess)
  vals .= 2obj_weight
  if length(y) > 0
    vals[1] -= 2y[1]
  end
  return rows, cols, vals
end

function NLPModels.hess_coord(nlp :: HS11, x :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hess)
  w = length(y) == 0 ? 0.0 : y[1]
  return ([1, 2], [1, 2], 2obj_weight * ones(2) + [-2w; 0.0])
end

function NLPModels.hprod!(nlp :: HS11, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hprod)
  nlp.counters.neval_hess -= 1
  Hx = Symmetric(hess(nlp, x, obj_weight=obj_weight, y=y), :L)
  Hv .= Hx * v
  return Hv
end

function NLPModels.cons!(nlp :: HS11, x :: AbstractVector, cx :: AbstractVector)
  increment!(nlp, :neval_cons)
  cx .= [-x[1]^2 + x[2]]
  return cx
end

function NLPModels.jac(nlp :: HS11, x :: AbstractVector)
  increment!(nlp, :neval_jac)
  return [-2 * x[1]  1.0]
end

function NLPModels.jac_structure!(nlp :: HS11, rows :: AbstractVector{Int}, cols :: AbstractVector{Int})
  rows[1:2] .= [1, 1]
  cols[1:2] .= [1, 2]
  return rows, cols
end

function NLPModels.jac_coord!(nlp :: HS11, x :: AbstractVector, rows :: AbstractVector{Int}, cols :: AbstractVector{Int}, vals :: AbstractVector)
  increment!(nlp, :neval_jac)
  vals .= [-2 * x[1], 1.0]
  return rows, cols, vals
end

function NLPModels.jac_coord(nlp :: HS11, x :: AbstractVector)
  increment!(nlp, :neval_jac)
  return ([1, 1], [1, 2], [-2 * x[1], 1.0])
end

function NLPModels.jprod!(nlp :: HS11, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jprod)
  Jv .= [-2 * x[1] * v[1] + v[2]]
  return Jv
end

function NLPModels.jtprod!(nlp :: HS11, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod)
  Jtv .= [-2 * x[1]; 1.0] * v[1]
  return Jtv
end
