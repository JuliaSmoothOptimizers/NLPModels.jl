# Brown and Dennis functions
#
#   Source: Problem 16 in
#   J.J. Moré, B.S. Garbow and K.E. Hillstrom,
#   "Testing Unconstrained Optimization Software",
#   ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981
#
#   classification SUR2-AN-4-0
using ForwardDiff

function brownden_autodiff()

  x0 = [25.0; 5.0; -5.0; -1.0]
  f(x) = begin
    s = 0.0
    for i = 1:20
      s += ((x[1] + x[2] * i/5 - exp(i/5))^2 + (x[3] + x[4] * sin(i/5) -
          cos(i/5))^2)^2
    end
    return s
  end

  return ADNLPModel(f, x0)
end

function browndenso_autodiff()

  x0 = [25.0; 5.0; -5.0; -1.0]
  fs = [x -> ((x[1] + x[2] * i/5 - exp(i/5))^2 + (x[3] + x[4] * sin(i/5) - cos(i/5))^2)^2 for i = 1:20]
  σfs = ones(20)

  return ADNLPModel(fs, σfs, x0)
end

function browndennls_autodiff()

  x0 = [25.0; 5.0; -5.0; -1.0]
  F(x) = [(x[1] + x[2] * i/5 - exp(i/5))^2 + (x[3] + x[4] * sin(i/5) - cos(i/5))^2 for i = 1:20]

  return ADNLPModel(F, 20, x0, σnls=2.0)
end

abstract type BROWNDENTypes <: AbstractNLPModel end

mutable struct BROWNDEN <: BROWNDENTypes
  meta :: NLPModelMeta
  counters :: Counters
end

mutable struct BROWNDENSO <: BROWNDENTypes
  meta :: NLPModelMeta
  σfs :: Array{Float64}
  counters :: Counters
end

mutable struct BROWNDENNLS <: BROWNDENTypes
  meta :: NLPModelMeta
  σnls :: Float64
  counters :: Counters
end

function BROWNDEN()
  meta = NLPModelMeta(4, nobjs=0, nlsequ=0, llsrows=0, x0=[25.0; 5.0; -5.0; -1.0], name="brownden")

  return BROWNDEN(meta, Counters())
end

function BROWNDENSO()
  meta = NLPModelMeta(4, nobjs=20, nlsequ=0, llsrows=0, x0=[25.0; 5.0; -5.0; -1.0], name="brownden")

  return BROWNDENSO(meta, ones(20), Counters())
end

function BROWNDENNLS()
  meta = NLPModelMeta(4, nobjs=0, nlsequ=20, llsrows=0, x0=[25.0; 5.0; -5.0; -1.0], name="brownden")

  return BROWNDENNLS(meta, 2.0, Counters())
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

function NLPModels.hess(nlp :: BROWNDEN, x :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hess)
  f(x) = sum(((x[1] + x[2] * i/5 - exp(i/5))^2 + (x[3] + x[4] * sin(i/5) - cos(i/5))^2)^2 for i = 1:20)
  return obj_weight * tril(ForwardDiff.hessian(f, x))
end

function NLPModels.hprod!(nlp :: BROWNDEN, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hprod)
  f(x) = sum(((x[1] + x[2] * i/5 - exp(i/5))^2 + (x[3] + x[4] * sin(i/5) - cos(i/5))^2)^2 for i = 1:20)
  Hv .= obj_weight * ForwardDiff.hessian(f, x) * v
  return Hv
end

function NLPModels.hprod(nlp :: BROWNDEN, x :: AbstractVector, v :: AbstractVector; obj_weight=1.0, y=Float64[])
  Hv = zeros(4)
  return hprod!(nlp, x, v, Hv, obj_weight=obj_weight)
end

function NLPModels.obj(nlp :: BROWNDENSO, i :: Int, x :: AbstractVector)
  @assert 1 ≤ i ≤ 20
  increment!(nlp, :neval_iobj)
  return ((x[1] + x[2] * i/5 - exp(i/5))^2 + (x[3] + x[4] * sin(i/5) - cos(i/5))^2)^2
end

function NLPModels.grad!(nlp :: BROWNDENSO, i :: Int, x :: AbstractVector, gx :: AbstractVector)
  @assert 1 ≤ i ≤ 20
  increment!(nlp, :neval_igrad)
  α(x,i) = x[1] + x[2] * i/5 - exp(i/5)
  β(x,i) = x[3] + x[4] * sin(i/5) - cos(i/5)
  θ(x,i) = α(x,i)^2 + β(x,i)^2
  gx .= 4 * θ(x,i) * (α(x,i) * [1; i/5; 0; 0] + β(x,i) * [0; 0; 1; sin(i/5)])
  return gx
end

function NLPModels.grad(nlp :: BROWNDENSO, i :: Int, x :: AbstractVector)
  @assert 1 ≤ i ≤ 20
  gx = zeros(4)
  return grad!(nlp, i, x, gx)
end

function NLPModels.hess(nlp :: BROWNDENSO, i :: Int, x :: AbstractVector)
  increment!(nlp, :neval_ihess)
  return tril(ForwardDiff.hessian(x -> obj(nlp, i, x), x))
end

function NLPModels.hprod!(nlp :: BROWNDENSO, i :: Int, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector)
  increment!(nlp, :neval_ihprod)
  Hv .= ForwardDiff.hessian(x -> obj(nlp, i, x), x) * v
  return Hv
end

function NLPModels.hprod(nlp :: BROWNDENSO, i :: Int, x :: AbstractVector, v :: AbstractVector)
  Hv = zeros(4)
  return hprod!(nlp, i, x, v, Hv)
end

function NLPModels.residual!(nlp :: BROWNDENNLS, x :: AbstractVector, Fx :: AbstractVector)
  increment!(nlp, :neval_residual)
  Fx .= [(x[1] + x[2] * i/5 - exp(i/5))^2 + (x[3] + x[4] * sin(i/5) - cos(i/5))^2 for i = 1:20]
  return Fx
end

function NLPModels.residual(nlp :: BROWNDENNLS, x :: AbstractVector)
  Fx = zeros(20)
  return residual!(nlp, x, Fx)
end

function NLPModels.jac_residual(nlp :: BROWNDENNLS, x :: AbstractVector)
  increment!(nlp, :neval_jac_residual)
  F(x) = [(x[1] + x[2] * i/5 - exp(i/5))^2 + (x[3] + x[4] * sin(i/5) - cos(i/5))^2 for i = 1:20]
  return ForwardDiff.jacobian(F, x)
end

function NLPModels.jprod_residual!(nlp :: BROWNDENNLS, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jprod_residual)
  F(x) = [(x[1] + x[2] * i/5 - exp(i/5))^2 + (x[3] + x[4] * sin(i/5) - cos(i/5))^2 for i = 1:20]
  Jv .= ForwardDiff.jacobian(F, x) * v
  return Jv
end

function NLPModels.jprod_residual(nlp :: BROWNDENNLS, x :: AbstractVector, v :: AbstractVector)
  Jv = zeros(20)
  return jprod_residual!(nlp, x, v, Jv)
end

function NLPModels.jtprod_residual!(nlp :: BROWNDENNLS, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod_residual)
  F(x) = [(x[1] + x[2] * i/5 - exp(i/5))^2 + (x[3] + x[4] * sin(i/5) - cos(i/5))^2 for i = 1:20]
  Jtv .= ForwardDiff.jacobian(F, x)' * v
  return Jtv
end

function NLPModels.jtprod_residual(nlp :: BROWNDENNLS, x :: AbstractVector, v :: AbstractVector)
  Jtv = zeros(4)
  return jtprod_residual!(nlp, x, v, Jtv)
end

function NLPModels.hess_residual(nlp :: BROWNDENNLS, x :: AbstractVector, i :: Int)
  @assert 1 ≤ i ≤ 20
  increment!(nlp, :neval_hess_residual)
  Fi(x) = (x[1] + x[2] * i/5 - exp(i/5))^2 + (x[3] + x[4] * sin(i/5) - cos(i/5))^2
  return tril(ForwardDiff.hessian(Fi, x))
end

function NLPModels.hprod_residual!(nlp :: BROWNDENNLS, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  @assert 1 ≤ i ≤ 20
  increment!(nlp, :neval_hess_residual)
  Fi(x) = (x[1] + x[2] * i/5 - exp(i/5))^2 + (x[3] + x[4] * sin(i/5) - cos(i/5))^2
  Hiv .= ForwardDiff.hessian(Fi, x) * v
  return Hiv
end
