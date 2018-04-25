# Convert an AbstractNLPModel to a MathProgBase model that can be passed
# to a MathProgBase standard solver.
# Modeled after Miles Lubin's AmplMathProgInterface

import MathProgBase
import MathProgBase.SolverInterface

export NLPModelEvaluator, NLPtoMPB


mutable struct NLPModelEvaluator{T <: AbstractNLPModel} <: SolverInterface.AbstractNLPEvaluator
  nlp :: T
end


MathProgBase.initialize(::NLPModelEvaluator, requested_features) = nothing

# SAutodetect?
MathProgBase.features_available(::NLPModelEvaluator) = [:Grad, :Jac, :Hess]

MathProgBase.eval_f(d::NLPModelEvaluator, x) = obj(d.nlp, x)

# use grad! ?
MathProgBase.eval_grad_f(d::NLPModelEvaluator, g, x) = copy!(g, grad(d.nlp, x))

MathProgBase.eval_g(d::NLPModelEvaluator, g, x) = copy!(g, cons(d.nlp, x))

function MathProgBase.jac_structure(d::NLPModelEvaluator)
  rows, cols, _ = jac_coord(d.nlp, [0.317i for i = 1:d.nlp.meta.nvar])
  return rows, cols
end

function MathProgBase.eval_jac_g(d::NLPModelEvaluator, J, x)
  _, _, vals = jac_coord(d.nlp, x)
  copy!(J, vals)
end

# use jprod! ?
function MathProgBase.eval_jac_prod(d::NLPModelEvaluator, y, x, w)
  y = jprod(d.nlp, x, w)
end

# use jtprod! ?
function MathProgBase.eval_jac_prod_t(d::NLPModelEvaluator, y, x, w)
  y = jtprod(d.nlp, x, w)
end

function MathProgBase.hesslag_structure(d::NLPModelEvaluator)
  rows, cols, _ = hess_coord(d.nlp, [0.317i for i = 1:d.nlp.meta.nvar],
                             y=[0.618i for i = 1:d.nlp.meta.ncon])
  return rows, cols
end

function MathProgBase.eval_hesslag(d::NLPModelEvaluator, H, x, σ, μ)
  rows, cols, vals = hess_coord(d.nlp, x, y=μ, obj_weight=σ)
  copy!(H, vals)
end

function MathProgBase.eval_hesslag_prod(d::NLPModelEvaluator, h, x, v, σ, μ)
  hprod!(d.nlp, x, v, h, y=μ, obj_weight=σ)
end

MathProgBase.isobjlinear(d::NLPModelEvaluator) = d.nlp.meta.nlo == 0

MathProgBase.isobjquadratic(d::NLPModelEvaluator) = false

MathProgBase.isconstrlinear(d::NLPModelEvaluator, i::Integer) = i in d.nlp.meta.lin

function loadNLPModel!{T <: AbstractNLPModel}(m::MathProgBase.AbstractMathProgModel, nlp::T)
  MathProgBase.loadproblem!(m, nlp.meta.nvar, nlp.meta.ncon,
                            nlp.meta.lvar, nlp.meta.uvar,
                            nlp.meta.lcon, nlp.meta.ucon,
                            nlp.meta.minimize ? :Min : :Max,
                            NLPModelEvaluator(nlp))
  MathProgBase.setwarmstart!(m, nlp.meta.x0)

  nnlvar = max(nlp.meta.nlvc, nlp.meta.nlvo)
  narcvar = nlp.meta.nwv
  nlinvar = nlp.meta.nvar - (nnlvar + narcvar + nlp.meta.nbv + nlp.meta.niv)
  nbinvar = nlp.meta.nbv
  nintvar = nlp.meta.niv

  v = fill(:Cont, nlp.meta.nvar)
  # First populate Table 4
  varidx = 1
  for i = 1:(nlp.meta.nlvb - nlp.meta.nlvbi)
    varidx += 1
  end
  for i = 1:nlp.meta.nlvbi
    v[varidx] = :Int
    varidx += 1
  end
  for i = 1:(nlp.meta.nlvc - (nlp.meta.nlvb + nlp.meta.nlvci))
    varidx += 1
  end
  for i = 1:nlp.meta.nlvci
    v[varidx] = :Int
    varidx += 1
  end
  for i = 1:(nlp.meta.nlvo - (nlp.meta.nlvc + nlp.meta.nlvoi))
    varidx += 1
  end
  for i = 1:nlp.meta.nlvoi
    v[varidx] = :Int
    varidx += 1
  end
  # Now populate Table 3
  varidx += narcvar + nlinvar
  for i = 1:nbinvar
    v[varidx] = :Bin
    varidx += 1
  end
  for i = 1:nintvar
    v[varidx] = :Int
    varidx += 1
  end
  @assert varidx == nlp.meta.nvar + 1
  # Set variable types
  if any(vtype -> vtype == :Int || vtype == :Bin, v)
      MathProgBase.setvartype!(m, v)
  end
  return m
end

"""
    mp = NLPtoMPB(nlp, solver)

Return a `MathProgBase` model corresponding to an `AbstractNLPModel`.

#### Arguments
- `nlp::AbstractNLPModel`
- `solver::AbstractMathProgSolver` a solver instance, e.g., `IpoptSolver()`

Currently, all models are treated as nonlinear models.

#### Return values
The function returns a `MathProgBase` model `mpbmodel` such that it should
be possible to call

    MathProgBase.optimize!(mpbmodel)
"""
function NLPtoMPB{T <: AbstractNLPModel}(nlp :: T, solver :: MathProgBase.AbstractMathProgSolver)
  model = MathProgBase.NonlinearModel(solver)
  return loadNLPModel!(model, nlp)
end

# MathProgNLPModels are straightforward
# Note that this method doesn't appear to be shadowed by the parametric NLPtoMPB
function NLPtoMPB(model :: MathProgNLPModel, solver :: MathProgBase.AbstractMathProgSolver)
  mpbmodel = MathProgBase.NonlinearModel(solver)
  MathProgBase.loadproblem!(mpbmodel,
                            model.meta.nvar, model.meta.ncon,
                            model.meta.lvar, model.meta.uvar,
                            model.meta.lcon, model.meta.ucon,
                            model.meta.minimize ? :Min : :Max,
                            NLPModelEvaluator(model))
  MathProgBase.setwarmstart!(mpbmodel, model.meta.x0)
  return mpbmodel
end
