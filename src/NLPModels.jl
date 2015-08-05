module NLPModels

using JuMP
using MathProgBase

require(Pkg.dir("MathProgBase", "src", "NLP", "NLP.jl"))
using NLP  # Defines NLPModelMeta.

export NLPModel,
       obj, grad,
       hess_coord, hess, hprod, hprod!

type NLPModel
  # meta :: NLPModelMeta
  model :: Model
  eval :: JuMP.JuMPNLPEvaluator

  # Make room for the objective gradient.
  g :: Array{Float64}

  # Make room for the Lagrangian Hessian.
  hvals :: Array{Float64}

  # Make room for a Hessian-vector product.
  hv :: Array{Float64}
end

"Construct an `NLPModel` from a JuMP `Model`"
function NLPModel(m :: Model)

  eval = JuMP.JuMPNLPEvaluator(m, JuMP.prepConstrMatrix(m))

  # :JacVec is not yet available.
  # features = [:Grad, :Jac, :JacVec, :Hess, :HessVec, :ExprGraph]
  features = [:Grad, :Jac, :Hess, :HessVec, :ExprGraph]
  MathProgBase.initialize(eval, features)

  # Stuff that will go into NLPModelMeta.
  # x0 = m.colVal
  nvar = MathProgBase.numvar(m)

  nlin = length(eval.m.linconstr)
  nquad = length(eval.m.quadconstr)
  nln = length(eval.m.nlpdata.nlconstr)
  numconstr = nlin + nquad + nln

  nnzh = length(eval.hess_I)

  return NLPModel(m,
                  eval,
                  zeros(nvar),  # g
                  zeros(nnzh),  # hvals
                  zeros(nvar)   # hv
                  )
end

import Base.show
show(nlp :: NLPModel) = show(nlp.model)

"Evaluate the objective function of `nlp` at `x`"
function obj(nlp :: NLPModel, x :: Array{Float64})
  return MathProgBase.eval_f(nlp.eval, x)
end

# TODO: Move g out of NLPModel?
"Evaluate the gradient of the objective function at `x`"
function grad(nlp :: NLPModel, x :: Array{Float64})
  MathProgBase.eval_grad_f(nlp.eval, nlp.g, x)
  return nlp.g
end

"Evaluate the gradient of the objective function at `x` in place"
function grad!(nlp :: NLPModel, x :: Array{Float64}, g :: Array{Float64})
  MathProgBase.eval_grad_f(nlp.eval, g, x)
  return g
end

"Evaluate the Lagrangian Hessian at `(x,y)` in sparse coordinate format"
function hess_coord(nlp :: NLPModel, x :: Array{Float64}, y :: Array{Float64})
  MathProgBase.eval_hesslag(nlp.eval, nlp.hvals, x, 1.0, y)
  return (nlp.eval.hess_I, nlp.eval.hess_J, nlp.hvals)
end

"Evaluate the objective Hessian at `x` in sparse coordinate format"
function hess_coord(nlp :: NLPModel, x :: Array{Float64})
  ncon = MathProgBase.numconstr(nlp.model)  # TODO: store in NLPModel.meta
  return hess_coord(nlp, x, zeros(ncon))
end

"Evaluate the Lagrangian Hessian at `(x,y)` as a sparse matrix"
function hess(nlp :: NLPModel, x :: Array{Float64}, y :: Array{Float64})
  nvar = MathProgBase.numvar(nlp.model)
  return sparse(hess_coord(nlp, x, y)..., nvar, nvar)
end

"Evaluate the objective Hessian at `x` as a sparse matrix"
function hess(nlp :: NLPModel, x :: Array{Float64})
  nvar = MathProgBase.numvar(nlp.model)
  return sparse(hess_coord(nlp, x)..., nvar, nvar)
end

# TODO: Move hv out of NLPModel
"Evaluate the Lagrangian Hessian-vector product at `(x,y)`"
function hprod(nlp :: NLPModel, x :: Array{Float64}, y :: Array{Float64}, v :: Array{Float64})
  MathProgBase.eval_hesslag_prod(nlp.eval, nlp.hv, x, v, 1.0, y)
  return nlp.hv
end

"Evaluate the Lagrangian Hessian-vector product at `(x,y)` in place"
function hprod!(nlp :: NLPModel, x :: Array{Float64}, y :: Array{Float64}, v :: Array{Float64}, hv :: Array{Float64})
  MathProgBase.eval_hesslag_prod(nlp.eval, hv, x, v, 1.0, y)
  return hv
end

"Evaluate the objective Hessian-vector product at `(x,y)`"
function hprod(nlp :: NLPModel, x :: Array{Float64}, v :: Array{Float64})
  ncon = MathProgBase.numconstr(nlp.model)  # TODO: store in NLPModel.meta
  return hprod(nlp, x, zeros(ncon), v)
end

"Evaluate the objective Hessian-vector product at `(x,y)` in place"
function hprod!(nlp :: NLPModel, x :: Array{Float64}, v :: Array{Float64}, hv :: Array{Float64})
  ncon = MathProgBase.numconstr(nlp.model)  # TODO: store in NLPModel.meta
  return hprod!(nlp, x, zeros(ncon), v, hv)
end

end # module
