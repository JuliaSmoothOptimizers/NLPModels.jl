export FeasibilityModel

# TODO: Extend to handle bounds
"""
A viability model is created from a NLPModel of the form

    min f(x)
    s.t c(x) = 0

by defining the function F(x) = c(x). If the problem has
bounds on the variables or more constraints, an error
is thrown.
"""
type FeasibilityModel <: AbstractNLSModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  counters :: NLSCounters
  nlp :: AbstractNLPModel
end

function FeasibilityModel(nlp :: AbstractNLPModel; name=nlp.meta.name)
  if !equality_constrained(nlp)
    if unconstrained(nlp)
      throw(ErrorException("Can't handle unconstrained problem"))
    else
      throw(ErrorException("Can't handle inequalities"))
    end
  end

  m, n = nlp.meta.ncon, nlp.meta.nvar
  # TODO: What is copied?
  meta = NLPModelMeta(n, x0=nlp.meta.x0, name=name, lvar=nlp.meta.lvar,
                      uvar=nlp.meta.uvar)
  nls_meta = NLSMeta(m, n)

  return FeasibilityModel(meta, nls_meta, NLSCounters(), nlp)
end

function residual(nls :: FeasibilityModel, x :: AbstractVector)
  nls.counters.neval_residual += 1
  return cons(nls.nlp, x)
end

function residual!(nls :: FeasibilityModel, x :: AbstractVector, Fx :: AbstractVector)
  nls.counters.neval_residual += 1
  return cons!(nls.nlp, x, Fx)
end

function jac_residual(nls :: FeasibilityModel, x :: AbstractVector)
  nls.counters.neval_jac_residual += 1
  return jac(nls.nlp, x)
end

function jprod_residual(nls :: FeasibilityModel, x :: AbstractVector, v :: AbstractVector)
  nls.counters.neval_jprod_residual += 1
  return jprod(nls.nlp, x, v)
end

function jprod_residual!(nls :: FeasibilityModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  nls.counters.neval_jprod_residual += 1
  return jprod!(nls.nlp, x, v, Jv)
end

function jtprod_residual(nls :: FeasibilityModel, x :: AbstractVector, v :: AbstractVector)
  nls.counters.neval_jtprod_residual += 1
  return jtprod(nls.nlp, x, v)
end

function jtprod_residual!(nls :: FeasibilityModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  nls.counters.neval_jtprod_residual += 1
  return jtprod!(nls.nlp, x, v, Jtv)
end

function hess_residual(nls :: FeasibilityModel, x :: AbstractVector, i :: Int)
  nls.counters.neval_hess_residual += 1
  y = zeros(nls.nls_meta.nequ)
  y[i] = 1.0
  return hess(nls.nlp, x, obj_weight = 0.0, y=y)
end

function hprod_residual(nls :: FeasibilityModel, x :: AbstractVector, i :: Int, v :: AbstractVector)
  nls.counters.neval_hprod_residual += 1
  y = zeros(nls.nls_meta.nequ)
  y[i] = 1.0
  return hprod(nls.nlp, x, v, obj_weight = 0.0, y=y)
end

function hprod_residual!(nls :: FeasibilityModel, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  nls.counters.neval_hprod_residual += 1
  y = zeros(nls.nls_meta.nequ)
  y[i] = 1.0
  return hprod!(nls.nlp, x, v, Hiv, obj_weight = 0.0, y=y)
end
