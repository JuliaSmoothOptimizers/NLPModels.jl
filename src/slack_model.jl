export SlackModel,
       reset!,
       obj, grad, grad!,
       cons, cons!, jac_coord, jac, jprod, jprod!, jtprod, jtprod!,
       hess_coord, hess, hprod, hprod!


"""A model whose only inequality constraints are bounds.

Given a model, this type represents a second model in which slack variables are
introduced so as to convert linear and nonlinear inequality constraints to
equality constraints and bounds. More precisely, if the original model has the
form

    min f(x)  s.t.  cₗ ≤ c(x) ≤ cᵤ and l ≤ x ≤ u,

the new model appears to the user as

    min f(X)  s.t.  g(X) = 0 and L ≤ X ≤ U.

The unknowns X = (x, s) contain the original variables and slack variables s.
The latter are such that the new model has the general form

    min f(x)  s.t.  c(x) - s = 0, cₗ ≤ s ≤ cᵤ and l ≤ x ≤ u

although no slack variables are introduced for equality constraints.

The slack variables are implicitly ordered as [s(low), s(upp), s(rng)], where
`low`, `upp` and `rng` represent the indices of the constraints of the form
cₗ ≤ c(x) < ∞, -∞ < c(x) ≤ cᵤ and cₗ ≤ c(x) ≤ cᵤ, respectively.
"""
type SlackModel <: AbstractNLPModel
  meta :: NLPModelMeta
  model :: AbstractNLPModel
end


"Construct a `SlackModel` from another type of model."
function SlackModel(model :: AbstractNLPModel)

  ns = model.meta.ncon - length(model.meta.jfix)
  ns == 0 && (return model)

  jlow = model.meta.jlow
  jupp = model.meta.jupp
  jrng = model.meta.jrng

  # Don't introduce slacks for equality constraints!
  lvar = [model.meta.lvar ; model.meta.lcon[[jlow ; jupp ; jrng]]]  # l ≤ x  and  cₗ ≤ s
  uvar = [model.meta.uvar ; model.meta.ucon[[jlow ; jupp ; jrng]]]  # x ≤ u  and  s ≤ cᵤ
  lcon = zeros(model.meta.ncon)
  lcon[model.meta.jfix] = model.meta.lcon[model.meta.jfix]
  ucon = zeros(model.meta.ncon)
  ucon[model.meta.jfix] = model.meta.ucon[model.meta.jfix]

  meta = NLPModelMeta(
    model.meta.nvar + ns,
    x0=[model.meta.x0 ; zeros(ns)],
    lvar=lvar,
    uvar=uvar,
    ncon=model.meta.ncon,
    lcon=lcon,
    ucon=ucon,
    y0=model.meta.y0,
    nnzj=model.meta.nnzj + ns,
    nnzh=model.meta.nnzh,
    lin=model.meta.lin,
    nln=model.meta.nln,
  )

  return SlackModel(meta, model)
end

import Base.show
# TODO: improve this!
# show(nlp :: SlackModel) = show(nlp.model)

"Reset evaluation counters in `nlp`"
function reset!(nlp :: SlackModel)
  reset!(nlp.model.counters)
  return nlp
end

"Evaluate the objective function of `nlp` at `x`."
function obj(nlp :: SlackModel, x :: Array{Float64})
  # f(X) = f(x)
  return obj(nlp.model, x[1:nlp.model.meta.nvar])
end

"Evaluate the gradient of the objective function at `x`."
function grad(nlp :: SlackModel, x :: Array{Float64})
  g = zeros(nlp.meta.nvar)
  return grad!(nlp, x, g)
end

"Evaluate the gradient of the objective function at `x` in place."
function grad!(nlp :: SlackModel, x :: Array{Float64}, g :: Array{Float64})
  # ∇f(X) = [∇f(x) ; 0]
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  grad!(nlp.model, x[1:nlp.model.meta.nvar], g)
  g[n+1:n+ns] = 0
  return g
end

"Evaluate the constraints at `x`."
function cons(nlp :: SlackModel, x :: Array{Float64})
  c = zeros(nlp.meta.ncon)
  return cons!(nlp, x, c)
end

"Evaluate the constraints at `x` in place."
function cons!(nlp :: SlackModel, x :: Array{Float64}, c :: Array{Float64})
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  nlow = length(nlp.model.meta.jlow)
  nupp = length(nlp.model.meta.jupp)
  nrng = length(nlp.model.meta.jrng)
  cons!(nlp.model, x[1:n], c)
  c[nlp.model.meta.jlow] -= x[n+1:n+nlow]
  c[nlp.model.meta.jupp] -= x[n+nlow+1:n+nlow+nupp]
  c[nlp.model.meta.jrng] -= x[n+nlow+nupp+1:n+nlow+nupp+nrng]
  return c
end

"Evaluate the constraints Jacobian at `x` in sparse coordinate format."
function jac_coord(nlp :: SlackModel, x :: Array{Float64})
  # J(X) = [J(x)  -I]
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  jrows, jcols, jvals = jac_coord(nlp.model, x[1:n])
  jlow = nlp.model.meta.jlow
  jupp = nlp.model.meta.jupp
  jrng = nlp.model.meta.jrng
  return (collect([jrows ; jlow ; jupp ; jrng]),
          collect([jcols ; collect(n+1:nlp.meta.nvar)]),
          collect([jvals ; -ones(ns)]))
end

"Evaluate the constraints Jacobian at `x` as a sparse matrix."
function jac(nlp :: SlackModel, x :: Array{Float64})
  return sparse(jac_coord(nlp, x)..., nlp.meta.ncon, nlp.meta.nvar)
end

"Evaluate the Jacobian-vector product at `x`."
function jprod(nlp :: SlackModel, x :: Array{Float64}, v :: Array{Float64})
  jv = zeros(nlp.ncon)
  return jprod!(nlp, x, v, jv)
end

"Evaluate the Jacobian-vector product at `x` in place."
function jprod!(nlp :: SlackModel, x :: Array{Float64}, v :: Array{Float64}, jv :: Array{Float64})
  # J(X) V = [J(x)  -I] [vₓ] = J(x) vₓ - vₛ
  #                     [vₛ]
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  jprod!(nlp.model, x[1:n], v[1:n], jv)
  k = 1
  # use 3 loops to avoid forming [jlow ; jupp ; jrng]
  for j in nlp.model.meta.jlow
    jv[j] -= v[n+k]
    k += 1
  end
  for j in nlp.model.meta.jupp
    jv[j] -= v[n+k]
    k += 1
  end
  for j in nlp.model.meta.jrng
    jv[j] -= v[n+k]
    k += 1
  end
  return jv
end

"Evaluate the transposed-Jacobian-vector product at `x`."
function jtprod(nlp :: SlackModel, x :: Array{Float64}, v :: Array{Float64})
  jtv = zeros(nlp.nvar)
  return jtprod!(nlp, x, v, jtv)
end

"Evaluate the transposed-Jacobian-vector product at `x` in place."
function jtprod!(nlp :: SlackModel, x :: Array{Float64}, v :: Array{Float64}, jtv :: Array{Float64})
  # J(X)ᵀ v = [J(x)ᵀ] v = [J(x)ᵀ v]
  #           [ -I  ]     [  -v   ]
  n = nlp.model.meta.nvar
  nlow = length(nlp.model.meta.jlow)
  nupp = length(nlp.model.meta.jupp)
  nrng = length(nlp.model.meta.jrng)
  jtprod!(nlp.model, x[1:n], v, jtv)
  jtv[n+1:n+nlow] = -v[nlp.model.meta.jlow]
  jtv[n+nlow+1:n+nlow+nupp] = -v[nlp.model.meta.jupp]
  jtv[n+nlow+nupp:nlp.nvar] = -v[nlp.model.meta.jrng]
  return jtv
end

"""Evaluate the Lagrangian Hessian at `(x,y)` in sparse coordinate format.
Only the lower triangle is returned.
"""
function hess_coord(nlp :: SlackModel, x :: Array{Float64}; σ :: Float64=1.0, y :: Array{Float64}=zeros(nlp.meta.ncon))
  # ∇²f(X) = [∇²f(x)  0]
  #          [0       0]
  n = nlp.model.meta.nvar
  return hess_coord(nlp.model, x[1:n], σ=σ, y=y)
end

"""Evaluate the Lagrangian Hessian at `(x,y)` as a sparse matrix.
Only the lower triangle is returned.
"""
function hess(nlp :: SlackModel, x :: Array{Float64}; σ :: Float64=1.0, y :: Array{Float64}=zeros(nlp.meta.ncon))
  return sparse(hess_coord(nlp, x, y=y, σ=σ)..., nlp.meta.nvar, nlp.meta.nvar)
end

"Evaluate the product of the Lagrangian Hessian at `(x,y)` with the vector `v`."
function hprod(nlp :: SlackModel, x :: Array{Float64}, v :: Array{Float64}; σ :: Float64=1.0, y :: Array{Float64}=zeros(nlp.meta.ncon))
  # ∇²f(X) V = [∇²f(x)  0] [vₓ ] = [∇²f(x) vₓ]
  #            [0       0] [vₛ]   [    0    ]
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  hv = zeros(nlp.meta.nvar)
  return hprod!(nlp, x, v, hv, σ=σ, y=y)
end

"Evaluate the product of the Lagrangian Hessian at `(x,y)` with the vector `v` in place."
function hprod!(nlp :: SlackModel, x :: Array{Float64}, v :: Array{Float64}, hv :: Array{Float64}; σ :: Float64=1.0, y :: Array{Float64}=zeros(nlp.meta.ncon))
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  # using hv[1:n] doesn't seem to work here
  hprod!(nlp.model, x[1:n], v[1:n], hv, σ=σ, y=y)
  hv[n+1:nlp.meta.nvar] = 0
  return hv
end
