using NLPModels: increment!

#Problem 10 in the Hock-Schittkowski suite
function hs10_autodiff()

  x0 = [-10.0; 10.0]
  f(x) = x[1] - x[2]
  c(x) = [-3 * x[1]^2 + 2 * x[1] * x[2] - x[2]^2 + 1.0]
  lcon = [0.0]
  ucon = [Inf]

  return ADNLPModel(f, x0, c=c, lcon=lcon, ucon=ucon)
end

mutable struct HS10 <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
end

function HS10()
  meta = NLPModelMeta(2, nobjs=0, nlsequ=0, llsrows=0, ncon=1,
                      x0=[-10.0; 10.0], lcon=[0.0], ucon=[Inf], name="HS10")

  return HS10(meta, Counters())
end

function NLPModels.obj(nlp :: HS10, x :: AbstractVector)
  increment!(nlp, :neval_obj)
  return x[1] - x[2]
end

function NLPModels.grad!(nlp :: HS10, x :: AbstractVector, gx :: AbstractVector)
  increment!(nlp, :neval_grad)
  gx .= [1.0; -1.0]
  return gx
end

function NLPModels.hess(nlp :: HS10, x :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hess)
  if length(y) > 0
    return y[1] * [-6.0  0.0; 2.0  -2.0]
  else
    return spzeros(2, 2)
  end
end

function NLPModels.hess_coord(nlp :: HS10, x :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hess)
  if length(y) > 0
    return ([1, 2, 2], [1, 1, 2], [-6.0, 2.0, -2.0] * y[1])
  else
    return (Int[], Int[], Float64[])
  end
end

function NLPModels.hprod!(nlp :: HS10, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hprod)
  if length(y) > 0
    Hv .= y[1] * [-6.0 * v[1] + 2.0 * v[2]; 2.0 * v[1] - 2.0 * v[2]]
  else
    fill!(Hv, 0.0)
  end
  return Hv
end

function NLPModels.hprod(nlp :: HS10, x :: AbstractVector, v :: AbstractVector; obj_weight=1.0, y=Float64[])
  Hv = zeros(nvar(nlp))
  return hprod!(nlp, x, v, Hv, obj_weight=obj_weight, y=y)
end

function NLPModels.cons!(nlp :: HS10, x :: AbstractVector, cx :: AbstractVector)
  increment!(nlp, :neval_cons)
  cx .= [-3 * x[1]^2 + 2 * x[1] * x[2] - x[2]^2 + 1.0]
  return cx
end

function NLPModels.cons(nlp :: HS10, x :: AbstractVector)
  cx = zeros(1)
  return cons!(nlp, x, cx)
end

function NLPModels.jac(nlp :: HS10, x :: AbstractVector)
  increment!(nlp, :neval_jac)
  return [-6 * x[1] + 2 * x[2]   2 * x[1] - 2 * x[2]]
end

function NLPModels.jprod!(nlp :: HS10, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jprod)
  Jv .= [(-6 * x[1] + 2 * x[2]) * v[1] + (2 * x[1] - 2 * x[2]) * v[2]]
  return Jv
end

function NLPModels.jprod(nlp :: HS10, x :: AbstractVector, v :: AbstractVector)
  Jv = zeros(ncon(nlp))
  return jprod!(nlp, x, v, Jv)
end

function NLPModels.jtprod!(nlp :: HS10, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod)
  Jtv .= [-6 * x[1] + 2 * x[2];  2 * x[1] - 2 * x[2]] * v[1]
  return Jtv
end

function NLPModels.jtprod(nlp :: HS10, x :: AbstractVector, v :: AbstractVector)
  Jtv = zeros(nvar(nlp))
  return jtprod!(nlp, x, v, Jtv)
end

