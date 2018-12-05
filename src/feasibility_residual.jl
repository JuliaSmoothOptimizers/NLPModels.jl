export FeasibilityResidual

# TODO: Extend to handle bounds
"""
A feasibility residual model is created from a NLPModel of the form

    min f(x)
    s.t c(x) = 0

by defining the function F(x) = c(x). If the problem has
bounds on the variables or more constraints, an error
is thrown.
"""
mutable struct FeasibilityResidual <: AbstractNLPModel
  meta :: NLPModelMeta
  σnls :: Float64
  counters :: Counters
  nlp :: AbstractNLPModel
end

function FeasibilityResidual(nlp :: AbstractNLPModel; name=nlp.meta.name)
  if !equality_constrained(nlp)
    if unconstrained(nlp)
      throw(ErrorException("Can't handle unconstrained problem"))
    else
      throw(ErrorException("Can't handle inequalities"))
    end
  end

  m, n = nlp.meta.ncon, nlp.meta.nvar
  # TODO: What is copied?
  meta = NLPModelMeta(n, nlsequ=m, x0=nlp.meta.x0, name=name, lvar=nlp.meta.lvar,
                      uvar=nlp.meta.uvar)
  feasnlp = FeasibilityResidual(meta, 1.0, Counters(), nlp)
  finalizer(nlp -> finalize(feasnlp.nlp), feasnlp)

  return feasnlp
end

function residual(nlp :: FeasibilityResidual, x :: AbstractVector)
  increment!(nlp, :neval_residual)
  return cons(nlp.nlp, x)
end

function residual!(nlp :: FeasibilityResidual, x :: AbstractVector, Fx :: AbstractVector)
  increment!(nlp, :neval_residual)
  return cons!(nlp.nlp, x, Fx)
end

function jac_residual(nlp :: FeasibilityResidual, x :: AbstractVector)
  increment!(nlp, :neval_jac_residual)
  return jac(nlp.nlp, x)
end

function jprod_residual(nlp :: FeasibilityResidual, x :: AbstractVector, v :: AbstractVector)
  increment!(nlp, :neval_jprod_residual)
  return jprod(nlp.nlp, x, v)
end

function jprod_residual!(nlp :: FeasibilityResidual, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jprod_residual)
  return jprod!(nlp.nlp, x, v, Jv)
end

function jtprod_residual(nlp :: FeasibilityResidual, x :: AbstractVector, v :: AbstractVector)
  increment!(nlp, :neval_jtprod_residual)
  return jtprod(nlp.nlp, x, v)
end

function jtprod_residual!(nlp :: FeasibilityResidual, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod_residual)
  return jtprod!(nlp.nlp, x, v, Jtv)
end

function hess_residual(nlp :: FeasibilityResidual, x :: AbstractVector, i :: Int)
  increment!(nlp, :neval_hess_residual)
  return chess(nlp.nlp, i, x)
end

function hprod_residual(nlp :: FeasibilityResidual, x :: AbstractVector, i :: Int, v :: AbstractVector)
  increment!(nlp, :neval_hprod_residual)
  return jth_hprod(nlp.nlp, x, v, i)
end

function hprod_residual!(nlp :: FeasibilityResidual, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  increment!(nlp, :neval_hprod_residual)
  return jth_hprod!(nlp.nlp, x, v, i, Hiv)
end
