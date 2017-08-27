using ForwardDiff

export ADNLSModel,
       residual!, jac_residual, jprod_residual!, jtprod_residual!,
       jac_op_residual, hess_residual, hprod_residual!

type ADNLSModel <: AbstractNLSModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  counters :: NLSCounters

  # Function
  F :: Function
end

function ADNLSModel(F :: Function, x0 :: Vector, m :: Int;
                    name :: String = "Generic",
                    lvar :: Vector = fill(-Inf, length(x0)),
                    uvar :: Vector = fill(Inf, length(x0))
                   )
  nvar = length(x0)

  meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar)
  nls_meta = NLSMeta(m, nvar)

  return ADNLSModel(meta, nls_meta, NLSCounters(), F)
end

ADNLSModel(F :: Function, n :: Int, m :: Int) = ADNLSModel(F, zeros(n), m)

function residual!(nls :: ADNLSModel, x :: AbstractVector, Fx :: AbstractVector)
  nls.counters.neval_residual += 1
  Fx[:] = nls.F(x)
  return Fx
end

function jac_residual(nls :: ADNLSModel, x :: Vector)
  nls.counters.neval_jac_residual += 1
  return ForwardDiff.jacobian(nls.F, x)
end

function jprod_residual!(nls :: ADNLSModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  nls.counters.neval_jprod_residual += 1
  Jv[:] = ForwardDiff.jacobian(nls.F, x) * v
  return Jv
end

function jtprod_residual!(nls :: ADNLSModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  nls.counters.neval_jtprod_residual += 1
  Jtv[:] = ForwardDiff.jacobian(nls.F, x)' * v
  return Jtv
end

function hess_residual(nls :: ADNLSModel, x :: AbstractVector, i :: Int)
  nls.counters.neval_hess_residual += 1
  return tril(ForwardDiff.hessian(x->nls.F(x)[i], x))
end

function hprod_residual!(nls :: ADNLSModel, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  nls.counters.neval_hprod_residual += 1
  Hiv[:] = ForwardDiff.hessian(x->nls.F(x)[i], x) * v
  return Hiv
end
