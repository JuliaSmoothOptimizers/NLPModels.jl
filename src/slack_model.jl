export SlackModel, SlackNLSModel


@doc raw"""A model whose only inequality constraints are bounds.

Given a model, this type represents a second model in which slack variables are
introduced so as to convert linear and nonlinear inequality constraints to
equality constraints and bounds. More precisely, if the original model has the
form

```math
\begin{align*}
       \min_x \quad & f(x)\\
\mathrm{s.t.} \quad & c_L ≤ c(x) ≤ c_U,\\
                    &  ℓ  ≤  x   ≤  u,
\end{align*}
```

the new model appears to the user as

```math
\begin{align*}
       \min_X \quad & f(X)\\
\mathrm{s.t.} \quad & g(X) = 0,\\
                    & L ≤ X ≤ U.
\end{align*}
```

The unknowns ``X = (x, s)`` contain the original variables and slack variables
``s``. The latter are such that the new model has the general form

```math
\begin{align*}
       \min_x \quad & f(x)\\
\mathrm{s.t.} \quad & c(x) - s = 0,\\
                    & c_L ≤ s ≤ c_U,\\
                    &  ℓ  ≤ x ≤ u.
\end{align*}
```

although no slack variables are introduced for equality constraints.

The slack variables are implicitly ordered as `[s(low), s(upp), s(rng)]`, where
`low`, `upp` and `rng` represent the indices of the constraints of the form
``c_L ≤ c(x) < ∞``, ``-∞ < c(x) ≤ c_U`` and
``c_L ≤ c(x) ≤ c_U``, respectively.
"""
mutable struct SlackModel <: AbstractNLPModel
  meta :: NLPModelMeta
  model :: AbstractNLPModel
end

"""Like `SlackModel`, this model converts inequalities into equalities and bounds.
"""
mutable struct SlackNLSModel <: AbstractNLSModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  model :: AbstractNLPModel
end

function slack_meta(meta :: NLPModelMeta; name=meta.name * "-slack")
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
    name=name
  )
end

"Construct a `SlackModel` from another type of model."
function SlackModel(model :: AbstractNLPModel; name=model.meta.name * "-slack")
  model.meta.ncon == length(model.meta.jfix) && return model

  meta = slack_meta(model.meta, name=name)

  snlp = SlackModel(meta, model)
  finalizer(nlp -> finalize(nlp.model), snlp)

  return snlp
end

function SlackNLSModel(model :: AbstractNLSModel; name=model.meta.name * "-slack")
  ns = model.meta.ncon - length(model.meta.jfix)
  ns == 0 && return model

  meta = slack_meta(model.meta, name=name)
  nls_meta = NLSMeta(model.nls_meta.nequ,
                     model.meta.nvar + ns,
                     x0=[model.meta.x0; zeros(ns)],
                     nnzj=model.nls_meta.nnzj,
                     nnzh=model.nls_meta.nnzh
                    )

  snls = SlackNLSModel(meta, nls_meta, model)
  finalizer(nls -> finalize(nls.model), snls)

  return snls
end

const SlackModels = Union{SlackModel,SlackNLSModel}

import Base.show
# TODO: improve this!
# show(nlp :: SlackModel) = show(nlp.model)

# retrieve counters from underlying model
for counter in fieldnames(Counters)
  @eval begin
    $counter(nlp :: SlackModels) = $counter(nlp.model)
    export $counter
  end
end

for counter in fieldnames(NLSCounters)
  counter == :Counters && continue
  @eval begin
    $counter(nlp :: SlackNLSModel) = $counter(nlp.model)
    export $counter
  end
end

sum_counters(nlp :: SlackModels) = sum_counters(nlp.model)

nls_meta(nlp :: SlackNLSModel) = nlp.nls_meta

function increment!(nlp :: SlackModels, s :: Symbol)
  increment!(nlp.model, s)
end

function reset!(nlp :: SlackModels)
  reset!(nlp.model.counters)
  return nlp
end

function obj(nlp :: SlackModels, x :: AbstractVector)
  # f(X) = f(x)
  return obj(nlp.model, @view x[1:nlp.model.meta.nvar])
end

function grad!(nlp :: SlackModels, x :: AbstractVector, g :: AbstractVector)
  # ∇f(X) = [∇f(x) ; 0]
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  @views grad!(nlp.model, x[1:n], g[1:n])
  g[n+1:n+ns] .= 0
  return g
end

function objgrad!(nlp :: SlackModels, x :: Array{Float64}, g :: Array{Float64})
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  @views f, _ = objgrad!(nlp.model, x[1:n], g[1:n])
  g[n+1:n+ns] .= 0
  return f, g
end

function cons!(nlp :: SlackModels, x :: AbstractVector, c :: AbstractVector)
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  nlow = length(nlp.model.meta.jlow)
  nupp = length(nlp.model.meta.jupp)
  nrng = length(nlp.model.meta.jrng)
  @views begin
    cons!(nlp.model, x[1:n], c)
    c[nlp.model.meta.jlow] -= x[n+1:n+nlow]
    c[nlp.model.meta.jupp] -= x[n+nlow+1:n+nlow+nupp]
    c[nlp.model.meta.jrng] -= x[n+nlow+nupp+1:n+nlow+nupp+nrng]
  end
  return c
end

function jac_structure!(nlp :: SlackModels, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  nnzj = nlp.model.meta.nnzj
  @views jac_structure!(nlp.model, rows[1:nnzj], cols[1:nnzj])
  jlow = nlp.model.meta.jlow
  jupp = nlp.model.meta.jupp
  jrng = nlp.model.meta.jrng
  nj, lj = nnzj, length(jlow)
  rows[nj+1:nj+lj] .= jlow
  nj, lj = nj + lj, length(jupp)
  rows[nj+1:nj+lj] .= jupp
  nj, lj = nj + lj, length(jrng)
  rows[nj+1:nj+lj] .= jrng
  cols[nnzj+1:end] .= n+1:nlp.meta.nvar
  return rows, cols
end

function jac_coord!(nlp :: SlackModels, x :: AbstractVector, vals :: AbstractVector)
  n = nlp.model.meta.nvar
  nnzj = nlp.model.meta.nnzj
  @views jac_coord!(nlp.model, x[1:n], vals[1:nnzj])
  vals[nnzj+1:nlp.meta.nnzj] .= -1
  return vals
end

function jprod!(nlp :: SlackModels, x :: AbstractVector, v :: AbstractVector, jv :: AbstractVector)
  # J(X) V = [J(x)  -I] [vₓ] = J(x) vₓ - vₛ
  #                     [vₛ]
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  @views jprod!(nlp.model, x[1:n], v[1:n], jv)
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

function jtprod!(nlp :: SlackModels, x :: AbstractVector, v :: AbstractVector, jtv :: AbstractVector)
  # J(X)ᵀ v = [J(x)ᵀ] v = [J(x)ᵀ v]
  #           [ -I  ]     [  -v   ]
  n = nlp.model.meta.nvar
  nlow = length(nlp.model.meta.jlow)
  nupp = length(nlp.model.meta.jupp)
  nrng = length(nlp.model.meta.jrng)
  @views begin
    jtprod!(nlp.model, x[1:n], v, jtv[1:n])
    jtv[n+1:n+nlow] = -v[nlp.model.meta.jlow]
    jtv[n+nlow+1:n+nlow+nupp] = -v[nlp.model.meta.jupp]
    jtv[n+nlow+nupp+1:nlp.meta.nvar] = -v[nlp.model.meta.jrng]
  end
  return jtv
end

function hess_structure!(nlp :: SlackModels, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  return hess_structure!(nlp.model, rows, cols)
end

function hess_coord!(nlp :: SlackModels, x :: AbstractVector, vals :: AbstractVector;
                     obj_weight :: Float64=1.0)
  n = nlp.model.meta.nvar
  return hess_coord!(nlp.model, view(x, 1:n), vals, obj_weight=obj_weight)
end

function hess_coord!(nlp :: SlackModels, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector;
                     obj_weight :: Float64=1.0)
  n = nlp.model.meta.nvar
  return hess_coord!(nlp.model, view(x, 1:n), y, vals, obj_weight=obj_weight)
end

# Kept in case some model implements `hess` but not `hess_coord/structure`
function hess(nlp :: SlackModels, x :: AbstractVector; kwargs...)
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  Hx = hess(nlp.model, view(x, 1:n); kwargs...)
  return [Hx spzeros(n, ns); spzeros(ns, n + ns)]
end

function hess(nlp :: SlackModels, x :: AbstractVector, y :: AbstractVector; kwargs...)
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  Hx = hess(nlp.model, view(x, 1:n), y; kwargs...)
  return [Hx spzeros(n, ns); spzeros(ns, n + ns)]
end

function hprod!(nlp :: SlackModels, x :: AbstractVector, v :: AbstractVector,
    hv :: AbstractVector;
    obj_weight :: Float64=1.0)
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  # using hv[1:n] doesn't seem to work here
  @views hprod!(nlp.model, x[1:n], v[1:n], hv[1:n], obj_weight=obj_weight)
  hv[n+1:nlp.meta.nvar] .= 0
  return hv
end

function hprod!(nlp :: SlackModels, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, hv :: AbstractVector; obj_weight :: Float64=1.0)
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  # using hv[1:n] doesn't seem to work here
  @views hprod!(nlp.model, x[1:n], y, v[1:n], hv[1:n], obj_weight=obj_weight)
  hv[n+1:nlp.meta.nvar] .= 0
  return hv
end

function residual!(nlp :: SlackNLSModel, x :: AbstractVector, Fx :: AbstractVector)
  return residual!(nlp.model, view(x, 1:nlp.model.meta.nvar), Fx)
end

function jac_residual(nlp :: SlackNLSModel, x :: AbstractVector)
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  ne = nlp.nls_meta.nequ
  Jx = jac_residual(nlp.model, @view x[1:n])
  if issparse(Jx)
    return [Jx spzeros(ne, ns)]
  else
    return [Jx zeros(ne, ns)]
  end
end

function jac_structure_residual!(nls :: SlackNLSModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  return jac_structure_residual!(nls.model, rows, cols)
end

function jac_coord_residual!(nls :: SlackNLSModel, x :: AbstractVector, vals :: AbstractVector)
  return jac_coord_residual!(nls.model, view(x, 1:nls.model.meta.nvar), vals)
end

function jprod_residual!(nlp :: SlackNLSModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  return jprod_residual!(nlp.model, view(x, 1:nlp.model.meta.nvar),
                         v[1:nlp.model.meta.nvar], Jv)
end

function jtprod_residual!(nlp :: SlackNLSModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  @views jtprod_residual!(nlp.model, x[1:n], v, Jtv[1:n])
  Jtv[n+1:n+ns] .= 0
  return Jtv
end

function jac_op_residual!(nls :: SlackNLSModel, x :: AbstractVector,
                          Jv :: AbstractVector, Jtv :: AbstractVector)
  prod = @closure v -> jprod_residual!(nls, x, v, Jv)
  ctprod = @closure v -> jtprod_residual!(nls, x, v, Jtv)
  return LinearOperator{Float64}(nls_meta(nls).nequ, nls_meta(nls).nvar,
                                 false, false, prod, ctprod, ctprod)
end

function hess_residual(nlp :: SlackNLSModel, x :: AbstractVector, v :: AbstractVector)
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  Hx = hess_residual(nlp.model, view(x, 1:n), v)
  if issparse(Hx)
    return [Hx spzeros(n, ns); spzeros(ns, n + ns)]
  else
    return [Hx zeros(n, ns); zeros(ns, n + ns)]
  end
end

function hess_structure_residual!(nls :: SlackNLSModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  return hess_structure_residual!(nls.model, rows, cols)
end

function hess_coord_residual!(nls :: SlackNLSModel, x :: AbstractVector, v :: AbstractVector, vals :: AbstractVector)
  return hess_coord_residual!(nls.model, view(x, 1:nls.model.meta.nvar), v, vals)
end

function jth_hess_residual(nlp :: SlackNLSModel, x :: AbstractVector, i :: Int)
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  Hx = jth_hess_residual(nlp.model, view(x, 1:n), i)
  if issparse(Hx)
    return [Hx spzeros(n, ns); spzeros(ns, n + ns)]
  else
    return [Hx zeros(n, ns); zeros(ns, n + ns)]
  end
end

function hprod_residual!(nlp :: SlackNLSModel, x :: AbstractVector, i :: Int, v :: AbstractVector, Hv :: AbstractVector)
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  @views hprod_residual!(nlp.model, x[1:n], i, v[1:n], Hv[1:n])
  Hv[n+1:n+ns] .= 0
  return Hv
end

function hess_op_residual!(nls :: SlackNLSModel, x :: AbstractVector, i :: Int, Hiv :: AbstractVector)
  prod = @closure v -> hprod_residual!(nls, x, i, v, Hiv)
  return LinearOperator{Float64}(nls_meta(nls).nvar, nls_meta(nls).nvar,
                                 true, true, prod, prod, prod)
end
