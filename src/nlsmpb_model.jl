using MathProgBase

include("mpb_model.jl")

export MathProgNLSModel,
       reset!,
       obj, grad, grad!,
       cons, cons!, jac_coord, jac, jprod, jprod!, jtprod, jtprod!,
       hess_coord, hess, hprod, hprod!,
       residual!, jac_residual, jprod_residual!, jtprod_residual!,
       jac_op_residual, hess_residual, hprod_residual!, cons, cons!, jac_coord,
       jac, jprod, jprod!, jtprod, jtprod!, hess, hess_coord, hprod, hprod!

mutable struct MathProgNLSModel <: AbstractNLSModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  Fmodel :: MathProgModel
  cmodel :: MathProgModel
  counters :: NLSCounters      # Evaluation counters.

  Fjrows :: Vector{Int}      # Jacobian sparsity pattern.
  Fjcols :: Vector{Int}
  Fjvals :: Vector{Float64}  # Room for the constraints Jacobian.

  Fhrows :: Vector{Int}      # Hessian sparsity pattern.
  Fhcols :: Vector{Int}
  Fhvals :: Vector{Float64}  # Room for the Lagrangian Hessian.

  cjrows :: Vector{Int}      # Jacobian sparsity pattern.
  cjcols :: Vector{Int}
  cjvals :: Vector{Float64}  # Room for the constraints Jacobian.

  chrows :: Vector{Int}      # Hessian sparsity pattern.
  chcols :: Vector{Int}
  chvals :: Vector{Float64}  # Room for the Lagrangian Hessian.
end

"Construct a `MathProgNLSModel` from two `MathProgModel`s."
function MathProgNLSModel(Fmodel :: MathProgModel,
                          cmodel :: MathProgModel;
                          name :: String="Generic")

  nvar = cmodel.numVar
  nequ = Fmodel.numConstr
  lvar = cmodel.lvar
  uvar = cmodel.uvar

  nlin = length(cmodel.eval.m.linconstr)         # Number of linear constraints.
  nquad = length(cmodel.eval.m.quadconstr)       # Number of quadratic constraints.
  nnln = length(cmodel.eval.m.nlpdata.nlconstr)  # Number of nonlinear constraints.
  ncon = cmodel.numConstr                        # Total number of constraints.
  lcon = cmodel.lcon
  ucon = cmodel.ucon

  Fjrows, Fjcols = MathProgBase.jac_structure(Fmodel.eval)
  cjrows, cjcols = MathProgBase.jac_structure(cmodel.eval)
  Fhrows, Fhcols = MathProgBase.hesslag_structure(Fmodel.eval)
  chrows, chcols = MathProgBase.hesslag_structure(cmodel.eval)

  meta = NLPModelMeta(nvar,
                      x0=cmodel.x,
                      lvar=lvar,
                      uvar=uvar,
                      ncon=ncon,
                      y0=zeros(ncon),
                      lcon=lcon,
                      ucon=ucon,
                      nnzj=length(cjrows),
                      nnzh=length(chrows),
                      lin=collect(1:nlin),  # linear constraints appear first in MPB
                      nln=collect(nlin+1:ncon),
                      name=name,
                      )

  return MathProgNLSModel(meta,
                          NLSMeta(nequ, nvar),
                          Fmodel,
                          cmodel,
                          NLSCounters(),
                          Fjrows,
                          Fjcols,
                          zeros(length(Fjrows)),  # Fjvals
                          Fhrows,
                          Fhcols,
                          zeros(length(Fhrows)),  # Fhvals
                          cjrows,
                          cjcols,
                          zeros(length(cjrows)),  # cjvals
                          chrows,
                          chcols,
                          zeros(length(chrows)),  # chvals
                         )
end

function residual!(nls :: MathProgNLSModel, x :: AbstractVector, Fx :: AbstractVector)
  increment!(nls, :neval_residual)
  MathProgBase.eval_g(nls.Fmodel.eval, Fx, x)
  return Fx
end

function jac_residual(nls :: MathProgNLSModel, x :: Vector)
  increment!(nls, :neval_jac_residual)
  m, n = nls.nls_meta.nequ, nls.meta.nvar
  MathProgBase.eval_jac_g(nls.Fmodel.eval, nls.Fjvals, x)
  return sparse(nls.Fjrows, nls.Fjcols, nls.Fjvals, m, n)
end

function jprod_residual!(nls :: MathProgNLSModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  nls.counters.neval_jac_residual -= 1
  increment!(nls, :neval_jprod_residual)
  Jv[:] = jac_residual(nls, x) * v
  return Jv
end

function jtprod_residual!(nls :: MathProgNLSModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  nls.counters.neval_jac_residual -= 1
  increment!(nls, :neval_jtprod_residual)
  Jtv[:] = jac_residual(nls, x)' * v
  return Jtv
end

function hess_residual(nls :: MathProgNLSModel, x :: AbstractVector, i :: Int)
  increment!(nls, :neval_hess_residual)
  y = [j == i ? 1.0 : 0.0 for j = 1:nls.nls_meta.nequ]
  n = nls.meta.nvar
  MathProgBase.eval_hesslag(nls.Fmodel.eval, nls.Fhvals, x, 0.0, y)
  return sparse(nls.Fhrows, nls.Fhcols, nls.Fhvals, n, n)
end

function hprod_residual!(nls :: MathProgNLSModel, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  increment!(nls, :neval_hprod_residual)
  y = [j == i ? 1.0 : 0.0 for j = 1:nls.nls_meta.nequ]
  MathProgBase.eval_hesslag_prod(nls.Fmodel.eval, Hiv, x, v, 0.0, y)
  return Hiv
end

function obj(nls :: MathProgNLSModel, x :: Array{Float64})
  increment!(nls, :neval_obj)
  return MathProgBase.eval_f(nls.cmodel.eval, x)
end

function grad(nls :: MathProgNLSModel, x :: Array{Float64})
  g = zeros(nls.meta.nvar)
  return grad!(nls, x, g)
end

function grad!(nls :: MathProgNLSModel, x :: Array{Float64}, g :: Array{Float64})
  increment!(nls, :neval_grad)
  MathProgBase.eval_grad_f(nls.cmodel.eval, g, x)
  return g
end

function cons(nls :: MathProgNLSModel, x :: Array{Float64})
  c = zeros(nls.meta.ncon)
  return cons!(nls, x, c)
end

function cons!(nls :: MathProgNLSModel, x :: Array{Float64}, c :: Array{Float64})
  increment!(nls, :neval_cons)
  MathProgBase.eval_g(nls.cmodel.eval, c, x)
  return c
end

function jac_coord(nls :: MathProgNLSModel, x :: Array{Float64})
  increment!(nls, :neval_jac)
  MathProgBase.eval_jac_g(nls.cmodel.eval, nls.cjvals, x)
  return (nls.cjrows, nls.cjcols, nls.cjvals)
end

function jac(nls :: MathProgNLSModel, x :: Array{Float64})
  return sparse(jac_coord(nls, x)..., nls.meta.ncon, nls.meta.nvar)
end

function jprod(nls :: MathProgNLSModel, x :: Array{Float64}, v :: Array{Float64})
  Jv = zeros(nls.meta.ncon)
  return jprod!(nls, x, v, Jv)
end

function jprod!(nls :: MathProgNLSModel,
                x :: Array{Float64},
                v :: Array{Float64},
                Jv :: Array{Float64})
  nls.counters.counters.neval_jac -= 1
  increment!(nls, :neval_jprod)
  Jv[:] = jac(nls, x) * v
  return Jv
end

function jtprod(nls :: MathProgNLSModel, x :: Array{Float64}, v :: Array{Float64})
  Jtv = zeros(nls.meta.nvar)
  return jtprod!(nls, x, v, Jtv)
end

function jtprod!(nls :: MathProgNLSModel,
                x :: Array{Float64},
                v :: Array{Float64},
                Jtv :: Array{Float64})
  nls.counters.counters.neval_jac -= 1
  increment!(nls, :neval_jtprod)
  Jtv[1:nls.meta.nvar] = jac(nls, x)' * v
  return Jtv
end

function hess_coord(nls :: MathProgNLSModel, x :: Array{Float64};
    obj_weight :: Float64=1.0, y :: Array{Float64}=zeros(nls.meta.ncon))
  increment!(nls, :neval_hess)
  MathProgBase.eval_hesslag(nls.cmodel.eval, nls.chvals, x, obj_weight, y)
  return (nls.chrows, nls.chcols, nls.chvals)
end

function hess(nls :: MathProgNLSModel, x :: Array{Float64};
    obj_weight :: Float64=1.0, y :: Array{Float64}=zeros(nls.meta.ncon))
  return sparse(hess_coord(nls, x, y=y, obj_weight=obj_weight)...,
                nls.meta.nvar, nls.meta.nvar)
end

function hprod(nls :: MathProgNLSModel, x :: Array{Float64}, v :: Array{Float64};
    obj_weight :: Float64=1.0, y :: Array{Float64}=zeros(nls.meta.ncon))
  hv = zeros(nls.meta.nvar)
  return hprod!(nls, x, v, hv, obj_weight=obj_weight, y=y)
end

#=
# Removed due to bug https://github.com/JuliaOpt/JuMP.jl/issues/1204
function hprod!(nls :: MathProgNLSModel, x :: Array{Float64}, v :: Array{Float64},
    hv :: Array{Float64};
    obj_weight :: Float64=1.0, y :: Array{Float64}=zeros(nls.meta.ncon))
  MathProgBase.eval_hesslag_prod(nls.cmodel.eval, hv, x, v, obj_weight, y)
  return hv
end
=#

function hprod!(nls :: MathProgNLSModel, x :: Array{Float64}, v :: Array{Float64},
    hv :: Array{Float64};
    obj_weight :: Float64=1.0, y :: Array{Float64}=zeros(nls.meta.ncon))
  increment!(nls, :neval_hprod)
  # See bug https://github.com/JuliaOpt/JuMP.jl/issues/1204
  MathProgBase.eval_hesslag_prod(nls.cmodel.eval, hv, x, v, 0.0, y)
  n = nls.meta.nvar
  if obj_weight != 0.0
    Fx = residual(nls, x)
    Jv = jprod_residual(nls, x, v)
    w = jtprod_residual(nls, x, Jv)
    hv[1:n] .+= w
    m = length(Fx)
    for i = 1:m
      hprod_residual!(nls, x, i, v, w)
      @views hv[1:n] .= hv[1:n] .+ Fx[i] * w
    end
    hv[1:n] .*= obj_weight
  end
  return hv
end
