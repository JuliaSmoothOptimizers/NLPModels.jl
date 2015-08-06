module NLPModels

using JuMP
using MathProgBase

require(Pkg.dir("MathProgBase", "src", "NLP", "NLP.jl"))
using NLP  # Defines NLPModelMeta.

export NLPModel,
       obj, grad,
       cons, cons!, jac_coord, jac, jprod, jprod!, jtprod, jtprod!,
       hess_coord, hess, hprod, hprod!

type NLPModel
  # meta :: NLPModelMeta
  model :: Model
  eval :: JuMP.JuMPNLPEvaluator

  g :: Array{Float64}      # Room for the objective gradient. 
  hvals :: Array{Float64}  # Room for the Lagrangian Hessian.
  hv :: Array{Float64}     # Room for a Hessian-vector product.
  c :: Array{Float64}      # Room for the constraints value.
  jvals :: Array{Float64}  # Room for the constraints Jacobian.
  jv :: Array{Float64}     # Room for a Jacobian-vector product.
  jtv :: Array{Float64}    # Room for a transposed-Jacobian-vector product.
end

"Construct an `NLPModel` from a JuMP `Model`"
function NLPModel(model :: Model; features=[:Grad, :Jac, :Hess, :HessVec, :ExprGraph])
  # TODO: :JacVec is not yet available.
  # features = [:Grad, :Jac, :JacVec, :Hess, :HessVec, :ExprGraph]

  eval = JuMP.JuMPNLPEvaluator(model, JuMP.prepConstrMatrix(model))
  MathProgBase.initialize(eval, features)

  # Stuff that will go into NLPModelMeta.
  x0 = model.colVal
  nvar = MathProgBase.numvar(model)
  lvar = model.colLower
  uvar = model.colUpper

  nlin = MathProgBase.numlinconstr(model)    # Number of linear constraints.
  nquad = MathProgBase.numquadconstr(model)  # Number of quadratic constraints.
  nln = length(eval.m.nlpdata.nlconstr)      # Number of nonlinear constraints.
  ncon = MathProgBase.numconstr(model)       # Total number of constraints.

  nnzj = length(eval.jac_I)
  nnzh = length(eval.hess_I)

  return NLPModel(model,
                  eval,
                  zeros(nvar),  # g
                  zeros(nnzh),  # hvals
                  zeros(nvar),  # hv
                  zeros(ncon),  # c
                  zeros(nnzj),  # jvals
                  zeros(ncon),  # jv
                  zeros(nvar),  # jtv
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

# TODO: Move c out of NLPModel?
"Evaluate the constraints at `x`"
function cons(nlp :: NLPModel, x :: Array{Float64})
  MathProgBase.eval_g(nlp.eval, nlp.c, x)
  return nlp.c
end

"Evaluate the constraints at `x` in place"
function cons!(nlp :: NLPModel, x :: Array{Float64}, c :: Array{Float64})
  MathProgBase.eval_g(nlp.eval, c, x)
  return c
end

"Evaluate the constraints Jacobian at `x` in sparse coordinate format"
function jac_coord(nlp :: NLPModel, x :: Array{Float64})
  MathProgBase.eval_jac_g(nlp.eval, nlp.jvals, x)
  return (nlp.eval.jac_I, nlp.eval.jac_J, nlp.jvals)
end

"Evaluate the constraints Jacobian at `x` as a sparse matrix"
function jac(nlp :: NLPModel, x :: Array{Float64})
  nvar = MathProgBase.numvar(nlp.model)  # TODO: store in NLPModel.meta
  ncon = MathProgBase.numconstr(nlp.model)
  return sparse(jac_coord(nlp, x)..., ncon, nvar)
end

"Evaluate the Jacobian-vector product at `x`"
function jprod(nlp :: NLPModel, x :: Array{Float64}, v :: Array{Float64})
  MathProgBase.eval_jac_prod(nlp.eval, nlp.jv, x, v)
  return nlp.jv
end

"Evaluate the Jacobian-vector product at `x` in place"
function jprod!(nlp :: NLPModel, x :: Array{Float64}, v :: Array{Float64}, jv ::
  Array{Float64})
  MathProgBase.eval_jac_prod(nlp.eval, jv, x, v)
  return jv
end

"Evaluate the transposed-Jacobian-vector product at `x`"
function jtprod(nlp :: NLPModel, x :: Array{Float64}, v :: Array{Float64})
  MathProgBase.eval_jac_prod_t(nlp.eval, nlp.jtv, x, v)
  return nlp.jtv
end

"Evaluate the transposed-Jacobian-vector product at `x` in place"
function jtprod!(nlp :: NLPModel, x :: Array{Float64}, v :: Array{Float64}, jtv ::
  Array{Float64})
  MathProgBase.eval_jac_prod_t(nlp.eval, jtv, x, v)
  return jtv
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
