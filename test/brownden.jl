# Brown and Dennis functions
#
#   Source: Problem 16 in
#   J.J. Moré, B.S. Garbow and K.E. Hillstrom,
#   "Testing Unconstrained Optimization Software",
#   ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981
#
#   classification SUR2-AN-4-0
using ForwardDiff

mutable struct BROWNDEN <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
end

function BROWNDEN()
  meta = NLPModelMeta(4, x0=[25.0; 5.0; -5.0; -1.0], name="brownden", nnzh=10)

  return BROWNDEN(meta, Counters())
end

function NLPModels.obj(nlp :: BROWNDEN, x :: AbstractVector)
  increment!(nlp, :neval_obj)
  return sum(((x[1] + x[2] * i/5 - exp(i/5))^2 + (x[3] + x[4] * sin(i/5) - cos(i/5))^2)^2 for i = 1:20)
end

function NLPModels.grad!(nlp :: BROWNDEN, x :: AbstractVector, gx :: AbstractVector)
  increment!(nlp, :neval_grad)
  α(x,i) = x[1] + x[2] * i/5 - exp(i/5)
  β(x,i) = x[3] + x[4] * sin(i/5) - cos(i/5)
  θ(x,i) = α(x,i)^2 + β(x,i)^2
  gx .= sum(4 * θ(x,i) * (α(x,i) * [1; i/5; 0; 0] + β(x,i) * [0; 0; 1; sin(i/5)]) for i = 1:20)
  return gx
end

function NLPModels.hess(nlp :: BROWNDEN, x :: AbstractVector; obj_weight=1.0, y=AbstractVector[])
  increment!(nlp, :neval_hess)
  f(x) = sum(((x[1] + x[2] * i/5 - exp(i/5))^2 + (x[3] + x[4] * sin(i/5) - cos(i/5))^2)^2 for i = 1:20)
  return obj_weight * tril(ForwardDiff.hessian(f, x))
end

function NLPModels.hess_structure!(nlp :: BROWNDEN, rows :: AbstractVector{Int}, cols :: AbstractVector{Int})
  n = nlp.meta.nvar
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows .= getindex.(I, 1)
  cols .= getindex.(I, 2)
  return rows, cols
end

function NLPModels.hess_coord!(nlp :: BROWNDEN, x :: AbstractVector, rows :: AbstractVector{Int}, cols :: AbstractVector{Int}, vals :: AbstractVector; obj_weight=1.0, y=AbstractVector[])
  Hx = hess(nlp, x, obj_weight=obj_weight, y=y)
  nnzh = length(vals)
  for k = 1:nnzh
    i, j = rows[k], cols[k]
    vals[k] = Hx[i,j]
  end
  return rows, cols, vals
end

function NLPModels.hess_coord(nlp :: BROWNDEN, x :: AbstractVector; obj_weight=1.0, y=AbstractVector[])
  Hx = hess(nlp, x, obj_weight=obj_weight, y=y)
  n = nlp.meta.nvar
  I = ((i,j,Hx[i,j]) for i = 1:n, j = 1:n if i ≥ j)
  return (getindex.(I, 1), getindex.(I, 2), getindex.(I, 3))
end

function NLPModels.hprod!(nlp :: BROWNDEN, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight=1.0, y=AbstractVector[])
  increment!(nlp, :neval_hprod)
  f(x) = sum(((x[1] + x[2] * i/5 - exp(i/5))^2 + (x[3] + x[4] * sin(i/5) - cos(i/5))^2)^2 for i = 1:20)
  Hv .= obj_weight * ForwardDiff.hessian(f, x) * v
  return Hv
end
