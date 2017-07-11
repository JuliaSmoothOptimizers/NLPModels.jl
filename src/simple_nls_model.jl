export SimpleNLSModel,
       residual, residual!, jac_residual, jprod_residual, jprod_residual!,
       jtprod_residual, jtprod_residual!, jac_op_residual, hess_residual,
       hprod_residual, hprod_residual!

"""
    nls = SimpleNLSModel(n;  F=F, F! =F!, J=J, Jp=Jp, Jp! =Jp!, Jtp=Jtp, Jtp! =Jtp!)
    nls = SimpleNLSModel(x0; F=F, F! =F!, J=J, Jp=Jp, Jp! =Jp!, Jtp=Jtp, Jtp! =Jtp!)

Creates a Nonlinear Linear Least Squares model to minimize ‖F(x)‖².
If J = J(x) is passed, the Jacobian is available.
"""
type SimpleNLSModel <: AbstractNLSModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  counters :: NLSCounters

  F :: Function
  F! :: Function
  J :: Function
  Jp :: Function
  Jp! :: Function
  Jtp :: Function
  Jtp! :: Function
  Hi :: Function # Hi(x, i) return the Hessian (tri-inf)
  Hip :: Function # Hip(x, i, v) returns the product
  Hip! :: Function # Hip!(x, i, v, Hiv) computes the product
end

NotImplemented(args...) = throw(NotImplementedError(""))

function SimpleNLSModel(x0 :: AbstractVector, nequ :: Int;
                        lvar :: Vector = fill(-Inf, length(x0)),
                        uvar :: Vector = fill(Inf, length(x0)),
                        F :: Function = NotImplemented,
                        F! :: Function = NotImplemented,
                        J :: Function = NotImplemented,
                        Jp :: Function = NotImplemented,
                        Jp! :: Function = NotImplemented,
                        Jtp :: Function = NotImplemented,
                        Jtp! :: Function = NotImplemented,
                        Hi :: Function = NotImplemented,
                        Hip :: Function = NotImplemented,
                        Hip! :: Function = NotImplemented
                       )
  meta = NLPModelMeta(length(x0), x0=x0)
  nls_meta = NLSMeta(nequ, length(x0))

  return SimpleNLSModel(meta, nls_meta, NLSCounters(), F, F!, J, Jp, Jp!, Jtp, Jtp!, Hi, Hip, Hip!)
end

SimpleNLSModel(nvar :: Int, nequ :: Int; kwargs...) = SimpleNLSModel(zeros(nvar), nequ; kwargs...)

function residual(nls :: SimpleNLSModel, x :: AbstractVector)
  nls.counters.neval_residual += 1
  return nls.F(x)
end

function residual!(nls :: SimpleNLSModel, x :: AbstractVector, Fx :: AbstractVector)
  nls.counters.neval_residual += 1
  nls.F!(x, Fx)
  return Fx
end

function jac_residual(nls :: SimpleNLSModel, x :: AbstractVector)
  nls.counters.neval_jac_residual += 1
  return nls.J(x)
end

function jprod_residual(nls :: SimpleNLSModel, x :: AbstractVector, v :: AbstractVector)
  nls.counters.neval_jprod_residual += 1
  return nls.Jp(x, v)
end

function jprod_residual!(nls :: SimpleNLSModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  nls.counters.neval_jprod_residual += 1
  nls.Jp!(x, v, Jv)
  return Jv
end

function jtprod_residual(nls :: SimpleNLSModel, x :: AbstractVector, v :: AbstractVector)
  nls.counters.neval_jtprod_residual += 1
  return nls.Jtp(x, v)
end

function jtprod_residual!(nls :: SimpleNLSModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  nls.counters.neval_jtprod_residual += 1
  nls.Jtp!(x, v, Jtv)
  return Jtv
end

function hess_residual(nls :: SimpleNLSModel, x :: AbstractVector, i :: Int)
  nls.counters.neval_hess_residual += 1
  return nls.Hi(x, i)
end

function hprod_residual(nls :: SimpleNLSModel, x :: AbstractVector, i :: Int, v :: AbstractVector)
  nls.counters.neval_hprod_residual += 1
  return nls.Hip(x, i, v)
end

function hprod_residual!(nls :: SimpleNLSModel, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  nls.counters.neval_hprod_residual += 1
  nls.Hip!(x, i, v, Hiv)
  return Hiv
end
