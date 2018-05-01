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

\\\\[ \\min f(x)  \\mbox{ s. t. }  c_L \\leq c(x) \\leq c_U \\mbox{ and }
\\ell \\leq x \\leq u, \\\\]

the new model appears to the user as

\\\\[ \\min f(X)  \\mbox{ s. t. }  g(X) = 0 \\mbox{ and } L \\leq X \\leq U. \\\\]

The unknowns \$X = (x, s)\$ contain the original variables and slack variables
\$s\$. The latter are such that the new model has the general form

\\\\[ \\min f(x)  \\mbox{ s. t. }  c(x) - s = 0, c_L \\leq s \\leq c_U \\mbox{ and }
\\ell \\leq x \\leq u, \\\\]

although no slack variables are introduced for equality constraints.

The slack variables are implicitly ordered as [s(low), s(upp), s(rng)], where
`low`, `upp` and `rng` represent the indices of the constraints of the form
\$c_L \\leq c(x) < \\infty\$, \$-\\infty < c(x) \\leq c_U\$ and
\$c_L \\leq c(x) \\leq c_U\$, respectively.
"""
mutable struct SlackModel <: AbstractNLPModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  model :: AbstractNLPModel
end

function slack_meta(meta :: NLPModelMeta)
  ns = meta.ncon - length(meta.jfix)
  jlow = meta.jlow
  jupp = meta.jupp
  jrng = meta.jrng

  # Don't introduce slacks for equality constraints!
  lvar = [meta.lvar ; meta.lcon[[jlow ; jupp ; jrng]]]  # l ≤ x  and  cₗ ≤ s
  uvar = [meta.uvar ; meta.ucon[[jlow ; jupp ; jrng]]]  # x ≤ u  and  s ≤ cᵤ
  lcon = zeros(meta.ncon)
  lcon[meta.jfix] = meta.lcon[meta.jfix]
  ucon = zeros(meta.ncon)
  ucon[meta.jfix] = meta.ucon[meta.jfix]

  return NLPModelMeta(
    meta.nvar + ns,
    x0=[meta.x0 ; zeros(ns)],
    lvar=lvar,
    uvar=uvar,
    ncon=meta.ncon,
    lcon=lcon,
    ucon=ucon,
    y0=meta.y0,
    nnzj=meta.nnzj + ns,
    nnzh=meta.nnzh,
    lin=meta.lin,
    nln=meta.nln,
  )
end

"Construct a `SlackModel` from another type of model."
function SlackModel(model :: AbstractNLPModel)
  model.meta.ncon == length(model.meta.jfix) && return model

  meta = slack_meta(model.meta)

  return SlackModel(meta, NLSMeta(0, 0), model)
end

function SlackModel(model :: AbstractNLSModel)
  ns = model.meta.ncon - length(model.meta.jfix)
  ns == 0 && return model

  meta = slack_meta(model.meta)
  nls_meta = NLSMeta(model.nls_meta.nequ,
                     model.meta.nvar + ns,
                     [model.meta.x0; zeros(ns)])

  return SlackModel(meta, nls_meta, model)
end

import Base.show
# TODO: improve this!
# show(nlp :: SlackModel) = show(nlp.model)

# retrieve counters from underlying model
for counter in fieldnames(Counters)
  @eval begin
    $counter(nlp :: SlackModel) = $counter(nlp.model)
    export $counter
  end
end

for counter in fieldnames(NLSCounters)
  counter == :Counters && continue
  @eval begin
    $counter(nlp :: SlackModel) = $counter(nlp.model)
    export $counter
  end
end

sum_counters(nlp :: SlackModel) = sum_counters(nlp.model)

nls_meta(nlp :: SlackModel) = nlp.nls_meta

function increment!(nlp :: SlackModel, s :: Symbol)
  increment!(nlp.model, s)
end

function reset!(nlp :: SlackModel)
  reset!(nlp.model.counters)
  return nlp
end

function obj(nlp :: SlackModel, x :: AbstractVector)
  # f(X) = f(x)
  return obj(nlp.model, x[1:nlp.model.meta.nvar])
end

function grad(nlp :: SlackModel, x :: AbstractVector)
  g = zeros(nlp.meta.nvar)
  return grad!(nlp, x, g)
end

function grad!(nlp :: SlackModel, x :: AbstractVector, g :: AbstractVector)
  # ∇f(X) = [∇f(x) ; 0]
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  grad!(nlp.model, x[1:n], g)
  @views g[n+1:n+ns] .= 0
  return g
end

function objgrad(nlp :: SlackModel, x :: Array{Float64})
  g = zeros(nlp.meta.nvar)
  return objgrad!(nlp, x, g)
end

function objgrad!(nlp :: SlackModel, x :: Array{Float64}, g :: Array{Float64})
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  f, _ = objgrad!(nlp.model, x[1:n], g)
  g[n+1:n+ns] = 0
  return f, g
end

function cons(nlp :: SlackModel, x :: AbstractVector)
  c = zeros(nlp.meta.ncon)
  return cons!(nlp, x, c)
end

function cons!(nlp :: SlackModel, x :: AbstractVector, c :: AbstractVector)
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  nlow = length(nlp.model.meta.jlow)
  nupp = length(nlp.model.meta.jupp)
  nrng = length(nlp.model.meta.jrng)
  cons!(nlp.model, x[1:n], c)
  @views c[nlp.model.meta.jlow] -= x[n+1:n+nlow]
  @views c[nlp.model.meta.jupp] -= x[n+nlow+1:n+nlow+nupp]
  @views c[nlp.model.meta.jrng] -= x[n+nlow+nupp+1:n+nlow+nupp+nrng]
  return c
end

function jac_coord(nlp :: SlackModel, x :: AbstractVector)
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

function jac(nlp :: SlackModel, x :: AbstractVector)
  return sparse(jac_coord(nlp, x)..., nlp.meta.ncon, nlp.meta.nvar)
end

function jprod(nlp :: SlackModel, x :: AbstractVector, v :: AbstractVector)
  jv = zeros(nlp.meta.ncon)
  return jprod!(nlp, x, v, jv)
end

function jprod!(nlp :: SlackModel, x :: AbstractVector, v :: AbstractVector, jv :: AbstractVector)
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

function jtprod(nlp :: SlackModel, x :: AbstractVector, v :: AbstractVector)
  jtv = zeros(nlp.meta.nvar)
  return jtprod!(nlp, x, v, jtv)
end

function jtprod!(nlp :: SlackModel, x :: AbstractVector, v :: AbstractVector, jtv :: AbstractVector)
  # J(X)ᵀ v = [J(x)ᵀ] v = [J(x)ᵀ v]
  #           [ -I  ]     [  -v   ]
  n = nlp.model.meta.nvar
  nlow = length(nlp.model.meta.jlow)
  nupp = length(nlp.model.meta.jupp)
  nrng = length(nlp.model.meta.jrng)
  jtprod!(nlp.model, x[1:n], v, jtv)
  @views jtv[n+1:n+nlow] = -v[nlp.model.meta.jlow]
  @views jtv[n+nlow+1:n+nlow+nupp] = -v[nlp.model.meta.jupp]
  @views jtv[n+nlow+nupp+1:nlp.meta.nvar] = -v[nlp.model.meta.jrng]
  return jtv
end

function hess_coord(nlp :: SlackModel, x :: AbstractVector;
    obj_weight :: Float64=1.0, y :: AbstractVector=zeros(nlp.meta.ncon))
  # ∇²f(X) = [∇²f(x)  0]
  #          [0       0]
  n = nlp.model.meta.nvar
  return hess_coord(nlp.model, x[1:n], obj_weight=obj_weight, y=y)
end

function hess(nlp :: SlackModel, x :: AbstractVector;
    obj_weight :: Float64=1.0, y :: AbstractVector=zeros(nlp.meta.ncon))
  return sparse(hess_coord(nlp, x, y=y, obj_weight=obj_weight)..., nlp.meta.nvar, nlp.meta.nvar)
end

function hprod(nlp :: SlackModel, x :: AbstractVector, v :: AbstractVector;
    obj_weight :: Float64=1.0, y :: AbstractVector=zeros(nlp.meta.ncon))
  # ∇²f(X) V = [∇²f(x)  0] [vₓ] = [∇²f(x) vₓ]
  #            [0       0] [vₛ]   [    0    ]
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  hv = zeros(nlp.meta.nvar)
  return hprod!(nlp, x, v, hv, obj_weight=obj_weight, y=y)
end

function hprod!(nlp :: SlackModel, x :: AbstractVector, v :: AbstractVector,
    hv :: AbstractVector;
    obj_weight :: Float64=1.0, y :: AbstractVector=zeros(nlp.meta.ncon))
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  # using hv[1:n] doesn't seem to work here
  hprod!(nlp.model, x[1:n], v[1:n], hv, obj_weight=obj_weight, y=y)
  @views hv[n+1:nlp.meta.nvar] .= 0
  return hv
end

# NLS API in case Slack in called on top of an NLSModel
function residual(nlp :: SlackModel, x :: AbstractVector)
  return residual(nlp.model, x[1:nlp.model.meta.nvar])
end

function residual!(nlp :: SlackModel, x :: AbstractVector, Fx :: AbstractVector)
  return residual!(nlp.model, x[1:nlp.model.meta.nvar], Fx)
end

function jac_residual(nlp :: SlackModel, x :: AbstractVector)
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  ne = nlp.nls_meta.nequ
  Jx = jac_residual(nlp.model, x[1:n])
  if issparse(Jx)
    return [Jx spzeros(ne, ns)]
  else
    return [Jx zeros(ne, ns)]
  end
end

function jprod_residual(nlp :: SlackModel, x :: AbstractVector, v :: AbstractVector)
  return jprod_residual(nlp.model, x[1:nlp.model.meta.nvar],
                        v[1:nlp.model.meta.nvar])
end

function jprod_residual!(nlp :: SlackModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  return jprod_residual!(nlp.model, x[1:nlp.model.meta.nvar],
                         v[1:nlp.model.meta.nvar], Jv)
end

function jtprod_residual(nlp :: SlackModel, x :: AbstractVector, v :: AbstractVector)
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  return [jtprod_residual(nlp.model, x[1:n], v); zeros(ns)]
end

function jtprod_residual!(nlp :: SlackModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  @views jtprod_residual!(nlp.model, x[1:n], v, Jtv[1:n])
  Jtv[n+1:n+ns] = 0
  return Jtv
end

function jac_op_residual(nls :: SlackModel, x :: AbstractVector)
  return LinearOperator{Float64}(nls_meta(nls).nequ, nls_meta(nls).nvar,
                                 false, false,
                                 v -> jprod_residual(nls, x, v),
                                 nothing,
                                 v -> jtprod_residual(nls, x, v))
end

function jac_op_residual!(nls :: SlackModel, x :: AbstractVector,
                          Jv :: AbstractVector, Jtv :: AbstractVector)
  return LinearOperator{Float64}(nls_meta(nls).nequ, nls_meta(nls).nvar,
                                 false, false,
                                 v -> jprod_residual!(nls, x, v, Jv),
                                 nothing,
                                 v -> jtprod_residual!(nls, x, v, Jtv))
end

function hess_residual(nlp :: SlackModel, x :: AbstractVector, i :: Int)
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  ne = nlp.nls_meta.nequ
  Hx = hess_residual(nlp.model, x[1:n], i)
  if issparse(Hx)
    return [Hx spzeros(ne, ns); spzeros(ns, ne + ns)]
  else
    return [Hx zeros(ne, ns); zeros(ns, ne + ns)]
  end
end

function hprod_residual(nlp :: SlackModel, x :: AbstractVector, i :: Int, v :: AbstractVector)
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  return [hprod_residual(nlp.model, x[1:n], i, v[1:n]); zeros(ns)]
end

function hprod_residual!(nlp :: SlackModel, x :: AbstractVector, i :: Int, v :: AbstractVector, Hv :: AbstractVector)
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  @views hprod_residual!(nlp.model, x[1:n], i, v[1:n], Hv[1:n])
  Hv[n+1:n+ns] = 0
  return Hv
end

function hess_op_residual(nls :: SlackModel, x :: AbstractVector, i :: Int)
  return LinearOperator(nls_meta(nls).nvar, nls_meta(nls).nvar,
                        true, true,
                        v -> hprod_residual(nls, x, i, v))
end

function hess_op_residual!(nls :: SlackModel, x :: AbstractVector, i :: Int, Hiv :: AbstractVector)
  return LinearOperator(nls_meta(nls).nvar, nls_meta(nls).nvar,
                        true, true,
                        v -> hprod_residual!(nls, x, i, v, Hiv))
end
