export FeasibilityResidual

# TODO: Extend to handle bounds
"""
A feasibility residual model is created from a NLPModel of the form

    min f(x)
    s.t cℓ ≤ c(x) ≤ cu
        bℓ ≤   x  ≤ bu

by creating slack variables s and defining F(x,s) = c(x) - s. The resulting NLS problem is

    min ¹/₂‖c(x) - s‖²
        bℓ ≤ x ≤ bu
        cℓ ≤ s ≤ bu

This is done using SlackModel first, and then defining the NLS. Notice that if bℓᵢ =
buᵢ, no slack variable is created.
"""
mutable struct FeasibilityResidual <: AbstractNLSModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  counters :: NLSCounters
  nlp :: AbstractNLPModel
end

function FeasibilityResidual(nlp :: AbstractNLPModel; name=nlp.meta.name)
  if !equality_constrained(nlp)
    if unconstrained(nlp)
      throw(ErrorException("Can't handle unconstrained problem"))
    else
      return FeasibilityResidual(SlackModel(nlp), name=name)
    end
  end

  m, n = nlp.meta.ncon, nlp.meta.nvar
  # TODO: What is copied?
  meta = NLPModelMeta(n, x0=nlp.meta.x0, name=name, lvar=nlp.meta.lvar,
                      uvar=nlp.meta.uvar, nnzj=0)
  nls_meta = NLSMeta(m, n, nnzj=nlp.meta.nnzj, nnzh=nlp.meta.nnzh)
  nls = FeasibilityResidual(meta, nls_meta, NLSCounters(), nlp)
  finalizer(nls -> finalize(nls.nlp), nls)

  return nls
end

function residual!(nls :: FeasibilityResidual, x :: AbstractVector, Fx :: AbstractVector)
  increment!(nls, :neval_residual)
  cons!(nls.nlp, x, Fx)
  Fx .-= nls.nlp.meta.lcon
  return Fx
end

function jac_residual(nls :: FeasibilityResidual, x :: AbstractVector)
  increment!(nls, :neval_jac_residual)
  return jac(nls.nlp, x)
end

function jac_structure_residual!(nls :: FeasibilityResidual, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  return jac_structure!(nls.nlp, rows, cols)
end

function jac_coord_residual!(nls :: FeasibilityResidual, x :: AbstractVector, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer}, vals :: AbstractVector)
  increment!(nls, :neval_jac_residual)
  return jac_coord!(nls.nlp, x, rows, cols, vals)
end

function jprod_residual!(nls :: FeasibilityResidual, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nls, :neval_jprod_residual)
  return jprod!(nls.nlp, x, v, Jv)
end

function jtprod_residual!(nls :: FeasibilityResidual, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nls, :neval_jtprod_residual)
  return jtprod!(nls.nlp, x, v, Jtv)
end

function hess_residual(nls :: FeasibilityResidual, x :: AbstractVector, v :: AbstractVector)
  increment!(nls, :neval_hess_residual)
  return hess(nls.nlp, x, obj_weight = 0.0, y=v)
end

function hess_structure_residual!(nls :: FeasibilityResidual, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  return hess_structure!(nls.nlp, rows, cols)
end

function hess_coord_residual!(nls :: FeasibilityResidual, x :: AbstractVector, v :: AbstractVector, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer}, vals :: AbstractVector)
  increment!(nls, :neval_hess_residual)
  return hess_coord!(nls.nlp, x, rows, cols, vals, obj_weight=0.0, y=v)
end

function jth_hess_residual(nls :: FeasibilityResidual, x :: AbstractVector, i :: Int)
  increment!(nls, :neval_jhess_residual)
  y = zeros(nls.nls_meta.nequ)
  y[i] = 1.0
  return hess(nls.nlp, x, obj_weight = 0.0, y=y)
end

function hprod_residual!(nls :: FeasibilityResidual, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  increment!(nls, :neval_hprod_residual)
  y = zeros(nls.nls_meta.nequ)
  y[i] = 1.0
  return hprod!(nls.nlp, x, v, Hiv, obj_weight = 0.0, y=y)
end

function hess(nls :: FeasibilityResidual, x :: AbstractVector;
              obj_weight :: Real=one(eltype(x)), y :: AbstractVector = eltype(x)[])
  increment!(nls, :neval_hess)
  cx = cons(nls.nlp, x)
  Jx = jac(nls.nlp, x)
  Hx = tril(Jx' * Jx)
  Hx .+= hess(nls.nlp, x, obj_weight=0.0, y=cx)
  return obj_weight * Hx
end

function hprod!(nls :: FeasibilityResidual, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector;
              obj_weight :: Real=one(eltype(x)), y :: AbstractVector = eltype(x)[])
  increment!(nls, :neval_hess)
  cx = cons(nls.nlp, x)
  Jv = jprod(nls.nlp, x, v)
  jtprod!(nls.nlp, x, Jv, Hv)
  Hiv = zeros(eltype(x), nls.meta.nvar)
  hprod!(nls.nlp, x, v, Hiv, obj_weight=0, y=cx)
  Hv .+= Hiv
  return obj_weight * Hv
end
