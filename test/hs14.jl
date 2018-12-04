#Problem 14 in the Hock-Schittkowski suite
function hs14_autodiff()

  x0 = [2.0; 2.0]
  f(x) = (x[1] - 2)^2 + (x[2] - 1)^2
  c(x) = [x[1] - 2 * x[2] + 1; -x[1]^2/4 - x[2]^2 + 1]
  lcon = [0.0; 0.0]
  ucon = [0.0; Inf]

  return ADNLPModel(f, x0, c=c, lcon=lcon, ucon=ucon)
end

mutable struct HS14 <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
end

function HS14()
  meta = NLPModelMeta(2, nobjs=0, nlsequ=0, llsrows=0, ncon=2,
                      x0=[2.0; 2.0], lcon=[0.0; 0.0], ucon=[0.0; Inf], name="HS14")

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

function NLPModels.hess_coord(nlp :: HS14, x :: AbstractVector; obj_weight=1.0, y=Float64[])
  Hx = hess(nlp, x, obj_weight=obj_weight, y=y)
  I = findall(!iszero, Hx)
  return (getindex.(I, 1), getindex.(I, 2), Hx[I])
end

function NLPModels.hprod!(nlp :: HS14, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hprod)
  nlp.counters.neval_hess -= 1
  Hx = Symmetric(hess(nlp, x, obj_weight=obj_weight, y=y), :L)
  Hv .= Hx * v
  return Hv
end

function NLPModels.hprod(nlp :: HS14, x :: AbstractVector, v :: AbstractVector; obj_weight=1.0, y=Float64[])
  Hv = zeros(nvar(nlp))
  return hprod!(nlp, x, v, Hv, obj_weight=obj_weight, y=y)
end

function NLPModels.cons!(nlp :: HS14, x :: AbstractVector, cx :: AbstractVector)
  increment!(nlp, :neval_cons)
  cx .= [x[1] - 2 * x[2] + 1; -x[1]^2/4 - x[2]^2 + 1]
  return cx
end

function NLPModels.cons(nlp :: HS14, x :: AbstractVector)
  cx = zeros(ncon(nlp))
  return cons!(nlp, x, cx)
end

function NLPModels.jac(nlp :: HS14, x :: AbstractVector)
  increment!(nlp, :neval_jac)
  return [1.0 -2.0; -x[1] / 2  -2 * x[2]]
end

function NLPModels.jprod!(nlp :: HS14, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jprod)
  Jv .= [v[1] - 2 * v[2]; -x[1] * v[1] / 2 - 2 * x[2] * v[2]]
  return Jv
end

function NLPModels.jprod(nlp :: HS14, x :: AbstractVector, v :: AbstractVector)
  Jv = zeros(ncon(nlp))
  return jprod!(nlp, x, v, Jv)
end

function NLPModels.jtprod!(nlp :: HS14, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod)
  Jtv .= [v[1] - x[1] * v[2] / 2; -2 * v[1] - 2 * x[2] * v[2]]
  return Jtv
end

function NLPModels.jtprod(nlp :: HS14, x :: AbstractVector, v :: AbstractVector)
  Jtv = zeros(nvar(nlp))
  return jtprod!(nlp, x, v, Jtv)
end

