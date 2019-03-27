using NLPModels: increment!

#Problem 6 in the Hock-Schittkowski suite
function hs6_autodiff()
  x0 = [-1.2; 1.0]
  f(x) = (1 - x[1])^2
  c(x) = [10 * (x[2] - x[1]^2)]
  lcon = [0.0]
  ucon = [0.0]

  return ADNLPModel(f, x0, c=c, lcon=lcon, ucon=ucon)
end

mutable struct HS6 <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
end

function HS6()
  meta = NLPModelMeta(2, ncon=1, nnzh=1, nnzj=2, x0=[-1.2; 1.0], lcon=[0.0], ucon=[0.0], name="hs6")

  return HS6(meta, Counters())
end

function NLPModels.obj(nlp :: HS6, x :: AbstractVector)
  increment!(nlp, :neval_obj)
  return (1 - x[1])^2
end

function NLPModels.grad!(nlp :: HS6, x :: AbstractVector, gx :: AbstractVector)
  increment!(nlp, :neval_grad)
  gx .= [2 * (x[1] - 1); 0.0]
  return gx
end

function NLPModels.hess(nlp :: HS6, x :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hess)
  w = length(y) > 0 ? y[1] : 0.0
  return [2.0 * obj_weight - 20 * w   0.0; 0.0 0.0]
end

function NLPModels.hess_structure!(nlp :: HS6, rows :: AbstractVector{Int}, cols :: AbstractVector{Int})
  rows[1] = 1
  cols[1] = 1
  return rows, cols
end

function NLPModels.hess_coord!(nlp :: HS6, x :: AbstractVector, rows :: AbstractVector{Int}, cols :: AbstractVector{Int}, vals :: AbstractVector; obj_weight=1.0, y=AbstractVector[])
  increment!(nlp, :neval_hess)
  vals[1] = 2.0 * obj_weight + (length(y) > 0 ? - 20 * y[1] : 0.0)
  return rows, cols, vals
end


function NLPModels.hess_coord(nlp :: HS6, x :: AbstractVector; obj_weight :: Real=1.0, y::AbstractVector=Float64[])
  increment!(nlp, :neval_hess)
  if length(y) > 0
    return ([1], [1], [2.0 * obj_weight - 20 * y[1]])
  else
    return ([1], [1], [2.0 * obj_weight])
  end
end

function NLPModels.hprod!(nlp :: HS6, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hprod)
  w = length(y) > 0 ? y[1] : 0.0
  Hv .= [(2.0 * obj_weight - 20 * w) * v[1]; 0.0]
  return Hv
end

function NLPModels.cons!(nlp :: HS6, x :: AbstractVector, cx :: AbstractVector)
  increment!(nlp, :neval_cons)
  cx[1] = 10 * (x[2] - x[1]^2)
  return cx
end

function NLPModels.jac(nlp :: HS6, x :: AbstractVector)
  increment!(nlp, :neval_jac)
  return [-20 * x[1]  10.0]
end

function NLPModels.jac_structure(nlp :: HS6)
  return ([1, 1], [1, 2])
end

function NLPModels.jac_coord!(nlp :: HS6, x :: AbstractVector, rows :: AbstractVector, cols :: AbstractVector, vals :: AbstractVector)
  increment!(nlp, :neval_jac)
  vals[1] = -20 * x[1]
  vals[2] = 10.0
  return rows, cols, vals
end

function NLPModels.jac_coord(nlp :: HS6, x :: AbstractVector)
  increment!(nlp, :neval_jac)
  return ([1, 1], [1, 2], [-20 * x[1], 10.0])
end

function NLPModels.jprod!(nlp :: HS6, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jprod)
  Jv .= [-20 * x[1] * v[1] + 10 * v[2]]
  return Jv
end

function NLPModels.jtprod!(nlp :: HS6, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod)
  Jtv .= [-20 * x[1]; 10] * v[1]
  return Jtv
end
