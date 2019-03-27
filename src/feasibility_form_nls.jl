export FeasibilityFormNLS,
       reset!,
       obj, grad, grad!,
       cons, cons!, jac_coord, jac, jprod, jprod!, jtprod, jtprod!,
       hess_coord, hess, hprod, hprod!


"""Converts a nonlinear least-squares problem with residual `F(x)` to a nonlinear
optimization problem with constraints `F(x) = r` and objective `¹/₂‖r‖²`. In other words,
converts

    min ¹/₂‖F(x)‖²
    s.t  cₗ ≤ c(x) ≤ cᵤ
          ℓ ≤   x  ≤ u

to

    min ¹/₂‖r‖²
    s.t   F(x) - r = 0
         cₗ ≤ c(x) ≤ cᵤ
          ℓ ≤   x  ≤ u

If you rather have the first problem, the `nls` model already works as an NLPModel of
that format.
"""
mutable struct FeasibilityFormNLS <: AbstractNLSModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  internal :: AbstractNLSModel
  counters :: NLSCounters
end

"""
    FeasibilityFormNLS(nls)

Converts a nonlinear least-squares problem with residual `F(x)` to a nonlinear
optimization problem with constraints `F(x) = r` and objective `¹/₂‖r‖²`.
"""
function FeasibilityFormNLS(nls :: AbstractNLSModel)
  nequ = nls.nls_meta.nequ
  meta = nls.meta
  nvar = meta.nvar + nequ
  ncon = meta.ncon + nequ
  nnzh = nls.nls_meta.nnzh + nequ + (meta.ncon == 0 ? 0 : meta.nnzh) # Some indexes can be repeated
  meta = NLPModelMeta(nvar, x0=[meta.x0; zeros(nequ)],
                      lvar=[meta.lvar; fill(-Inf, nequ)],
                      uvar=[meta.uvar; fill( Inf, nequ)],
                      ncon=ncon,
                      lcon=[zeros(nequ); meta.lcon],
                      ucon=[zeros(nequ); meta.ucon],
                      y0=[zeros(nequ); meta.y0],
                      lin=meta.lin,
                      nln=[1:nequ; (meta.nln) .+ nequ],
                      nnzj=meta.nnzj + nls.nls_meta.nnzj + nequ,
                      nnzh=nnzh
                     )
  nls_meta = NLSMeta(nequ, nvar, x0=[meta.x0; zeros(nequ)], nnzj=nequ, nnzh=0)

  nlp = FeasibilityFormNLS(meta, nls_meta, nls, NLSCounters())
  finalizer(nlp -> finalize(nlp.internal), nlp)

  return nlp
end

function FeasibilityFormNLS(nls :: FeasibilityResidual)
  meta = nls.nlp.meta
  nequ = meta.ncon
  nvar = meta.nvar + nequ
  ncon = meta.ncon
  nnzj = meta.nnzj + nequ
  nnzh = meta.nnzh + nequ
  meta = NLPModelMeta(nvar, x0=[meta.x0; zeros(nequ)],
                      lvar=[meta.lvar; fill(-Inf, nequ)],
                      uvar=[meta.uvar; fill( Inf, nequ)],
                      ncon=ncon,
                      lcon=meta.lcon,
                      ucon=meta.ucon,
                      y0=meta.y0,
                      lin=meta.lin,
                      nln=meta.nln,
                      nnzj=nnzj, nnzh=nnzh
                     )
  nls_meta = NLSMeta(nequ, nvar, x0=[meta.x0; zeros(nequ)], nnzj=nequ, nnzh=0)

  nlp = FeasibilityFormNLS(meta, nls_meta, nls, NLSCounters())
  finalizer(nlp -> finalize(nlp.internal), nlp)

  return nlp
end

function obj(nlp :: FeasibilityFormNLS, x :: AbstractVector)
  increment!(nlp, :neval_obj)
  n = nlp.internal.meta.nvar
  r = @view x[n+1:end]
  return dot(r, r) / 2
end

function grad(nlp :: FeasibilityFormNLS, x :: AbstractVector)
  g = zeros(nlp.meta.nvar)
  return grad!(nlp, x, g)
end

function grad!(nlp :: FeasibilityFormNLS, x :: AbstractVector, g :: AbstractVector)
  increment!(nlp, :neval_grad)
  n = nlp.internal.meta.nvar
  g[1:n] .= 0.0
  g[n+1:end] .= @view x[n+1:end]
  return g
end

function objgrad(nlp :: FeasibilityFormNLS, x :: Array{Float64})
  g = zeros(nlp.meta.nvar)
  return objgrad!(nlp, x, g)
end

function objgrad!(nlp :: FeasibilityFormNLS, x :: Array{Float64}, g :: Array{Float64})
  increment!(nlp, :neval_obj)
  increment!(nlp, :neval_grad)
  n = nlp.internal.meta.nvar
  r = @view x[n+1:end]
  f = dot(r, r) / 2
  g[1:n] .= 0.0
  g[n+1:end] .= @view x[n+1:end]
  return f, g
end

function cons(nlp :: FeasibilityFormNLS, x :: AbstractVector)
  c = zeros(nlp.meta.ncon)
  return cons!(nlp, x, c)
end

function cons!(nlp :: FeasibilityFormNLS, xr :: AbstractVector, c :: AbstractVector)
  increment!(nlp, :neval_cons)
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  x = @view xr[1:n]
  r = @view xr[n+1:end]
  residual!(nlp.internal, x, @view c[1:ne])
  c[1:ne] .-= r
  if m > 0
    cons!(nlp.internal, x, @view c[ne+1:end])
  end
  return c
end

function jac_structure(nlp :: FeasibilityFormNLS)
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  IF, JF = jac_structure_residual(nlp.internal)
  (Ic, Jc) = m > 0 ? jac_structure(nlp.internal) : (Int[], Int[])
  I = [IF; Ic .+ ne; 1:ne]
  J = [JF; Jc; (n+1):(n+ne)]
  return I, J
end

function jac_coord!(nlp :: FeasibilityFormNLS, xr :: AbstractVector, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer}, vals :: AbstractVector)
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  x = @view xr[1:n]
  nnzjF = nlp.internal.nls_meta.nnzj
  nnzjc = m > 0 ? nlp.internal.meta.nnzj : 0
  I = 1:nnzjF
  @views jac_coord_residual!(nlp.internal, x, rows[I], cols[I], vals[I])
  if m > 0
    I = nnzjF+1:nnzjF+nnzjc
    @views jac_coord!(nlp.internal, x, rows[I] .- ne, cols[I], vals[I])
  end
  vals[nnzjF+nnzjc+1:nnzjF+nnzjc+ne] .= -1
  return rows, cols, vals
end

function jac_coord(nlp :: FeasibilityFormNLS, xr :: AbstractVector)
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  x = @view xr[1:n]
  IF, JF, VF = jac_coord_residual(nlp.internal, x)
  (Ic, Jc, Vc) = m > 0 ? jac_coord(nlp.internal, x) : (Int[], Int[], Float64[])
  I = [IF; Ic .+ ne; 1:ne]
  J = [JF; Jc; (n+1):(n+ne)]
  V = [VF; Vc; -ones(ne)]
  return I, J, V
end

function jac(nlp :: FeasibilityFormNLS, xr :: AbstractVector)
  increment!(nlp, :neval_jac)
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  x = @view xr[1:n]
  JF = jac_residual(nlp.internal, x)
  JC = m > 0 ? jac(nlp.internal, x) : spzeros(m, n)
  return [JF -spdiagm(0 => ones(ne)); JC spzeros(m, ne)]
end

function jprod(nlp :: FeasibilityFormNLS, x :: AbstractVector, v :: AbstractVector)
  jv = zeros(nlp.meta.ncon)
  return jprod!(nlp, x, v, jv)
end

function jprod!(nlp :: FeasibilityFormNLS, xr :: AbstractVector, v :: AbstractVector, jv :: AbstractVector)
  increment!(nlp, :neval_jprod)
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  x = @view xr[1:n]
  @views jprod_residual!(nlp.internal, x, v[1:n], jv[1:ne])
  @views jv[1:ne] .-= v[n+1:end]
  if m > 0
    @views jprod!(nlp.internal, x, v[1:n], jv[ne+1:end])
  end
  return jv
end

function jtprod(nlp :: FeasibilityFormNLS, x :: AbstractVector, v :: AbstractVector)
  jtv = zeros(nlp.meta.nvar)
  return jtprod!(nlp, x, v, jtv)
end

function jtprod!(nlp :: FeasibilityFormNLS, xr :: AbstractVector, v :: AbstractVector, jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod)
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  x = @view xr[1:n]
  @views jtprod_residual!(nlp.internal, x, v[1:ne], jtv[1:n])
  if m > 0
    @views jtv[1:n] .+= jtprod(nlp.internal, x, v[ne+1:end])
  end
  @views jtv[n+1:end] .= -v[1:ne]
  return jtv
end

function hess_structure!(nlp :: FeasibilityFormNLS, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  nnzhF = nlp.internal.nls_meta.nnzh
  nnzhc = m > 0 ? nlp.internal.meta.nnzh : 0
  I = 1:nnzhF
  @views hess_structure_residual!(nlp.internal, rows[I], cols[I])
  if m > 0
    I = nnzhF+1:nnzhF+nnzhc
    @views hess_structure!(nlp.internal, rows[I], cols[I])
  end
  I = nnzhF+nnzhc+1:nnzhF+nnzhc+ne
  rows[I] .= n+1:n+ne
  cols[I] .= n+1:n+ne
  return rows, cols
end

function hess_coord!(nlp :: FeasibilityFormNLS, xr :: AbstractVector, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer}, vals :: AbstractVector;
    obj_weight :: Float64=1.0, y :: AbstractVector=zeros(nlp.meta.ncon))
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  nnzhF = nlp.internal.nls_meta.nnzh
  nnzhc = m > 0 ? nlp.internal.meta.nnzh : 0
  x = @view xr[1:n]
  y1 = @view y[1:ne]
  y2 = @view y[ne+1:ne+m]
  I = 1:nnzhF
  @views hess_coord_residual!(nlp.internal, x, y1, rows[I], cols[I], vals[I])
  if m > 0
    I = nnzhF+1:nnzhF+nnzhc
    @views hess_coord!(nlp.internal, x, rows[I], cols[I], vals[I], obj_weight=0.0, y=y2)
  end
  vals[nnzhF+nnzhc+1:nnzhF+nnzhc+ne] .= obj_weight
  return rows, cols, vals
end

function hess_coord(nlp :: FeasibilityFormNLS, xr :: AbstractVector;
    obj_weight :: Float64=1.0, y :: AbstractVector=zeros(nlp.meta.ncon))
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  x = @view xr[1:n]
  y1 = @view y[1:ne]
  y2 = @view y[ne+1:ne+m]
  IF, JF, VF = hess_coord_residual(nlp.internal, x, y1)
  (Ic, Jc, Vc) = m > 0 ? hess_coord(nlp.internal, x, obj_weight=0.0, y=y2) : (Int[], Int[], Float64[])
  I = [IF; Ic; (n+1:n+ne)]
  J = [JF; Jc; (n+1:n+ne)]
  V = [VF; Vc; obj_weight * ones(ne)]
  return I, J, V
end

function hess(nlp :: FeasibilityFormNLS, xr :: AbstractVector;
    obj_weight :: Float64=1.0, y :: AbstractVector=zeros(nlp.meta.ncon))
  increment!(nlp, :neval_hess)
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  x = @view xr[1:n]
  @views Hx = m > 0 ? hess(nlp.internal, x, obj_weight=0.0, y=y[ne+1:end]) : spzeros(n, n)
  Hx += hess_residual(nlp.internal, x, @view y[1:ne])
  return [Hx spzeros(n, ne); spzeros(ne, n) obj_weight * I]
end

function hprod(nlp :: FeasibilityFormNLS, x :: AbstractVector, v :: AbstractVector;
    obj_weight :: Float64=1.0, y :: AbstractVector=zeros(nlp.meta.ncon))
  hv = zeros(nlp.meta.nvar)
  return hprod!(nlp, x, v, hv, obj_weight=obj_weight, y=y)
end

function hprod!(nlp :: FeasibilityFormNLS, xr :: AbstractVector, v :: AbstractVector,
    hv :: AbstractVector;
    obj_weight :: Float64=1.0, y :: AbstractVector=zeros(nlp.meta.ncon))
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  x = @view xr[1:n]
  if m > 0
    @views hprod!(nlp.internal, x, v[1:n], hv[1:n], obj_weight=0.0, y=y[ne+1:end])
  else
    fill!(hv, 0.0)
  end
  for i = 1:ne
    @views hv[1:n] .+= hprod_residual(nlp.internal, x, i, v[1:n]) * y[i]
  end
  @views hv[n+1:end] .= obj_weight * v[n+1:end]
  return hv
end

function residual!(nlp :: FeasibilityFormNLS, x :: AbstractVector, Fx :: AbstractVector)
  increment!(nlp, :neval_residual)
  n = nlp.internal.meta.nvar
  Fx .= @view x[n+1:end]
  return Fx
end

function jac_residual(nlp :: FeasibilityFormNLS, x :: AbstractVector)
  increment!(nlp, :neval_jac_residual)
  n, ne = nlp.internal.meta.nvar, nlp.internal.nls_meta.nequ
  return [spzeros(ne, n) I]
end

function jac_structure_residual(nlp :: FeasibilityFormNLS)
  n, ne = nlp.internal.meta.nvar, nlp.internal.nls_meta.nequ
  return collect(1:ne), collect((n+1):(n+ne))
end

function jac_coord_residual!(nlp :: FeasibilityFormNLS, x :: AbstractVector, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer}, vals :: AbstractVector)
  increment!(nlp, :neval_jac_residual)
  vals[1:nlp.nls_meta.nnzj] .= 1
  return rows, cols, vals
end

function jac_coord_residual(nlp :: FeasibilityFormNLS, x :: AbstractVector)
  increment!(nlp, :neval_jac_residual)
  n, ne = nlp.internal.meta.nvar, nlp.internal.nls_meta.nequ
  return collect(1:ne), collect((n+1):(n+ne)), ones(ne)
end

function jprod_residual!(nlp :: FeasibilityFormNLS, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jprod_residual)
  n = nlp.internal.meta.nvar
  Jv .= @view v[n+1:end]
  return Jv
end

function jtprod_residual!(nlp :: FeasibilityFormNLS, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod_residual)
  n, ne = nlp.internal.meta.nvar, nlp.internal.nls_meta.nequ
  Jtv[1:n] .= 0.0
  Jtv[n+1:end] .= v
  return Jtv
end

function hess_residual(nlp :: FeasibilityFormNLS, x :: AbstractVector, v :: AbstractVector)
  increment!(nlp, :neval_hess_residual)
  n = nlp.meta.nvar
  return spzeros(n, n)
end

function hess_structure_residual!(nlp :: FeasibilityFormNLS, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  return rows, cols  # Hessian of residual is zero; do not change rows and cols
end

function hess_coord_residual!(nlp :: FeasibilityFormNLS, x :: AbstractVector, v :: AbstractVector, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer}, vals :: AbstractVector)
  increment!(nlp, :neval_hess_residual)
  return rows, cols, vals
end

function hess_coord_residual(nlp :: FeasibilityFormNLS, x :: AbstractVector, v :: AbstractVector)
  increment!(nlp, :neval_hess_residual)
  return (Int[], Int[], Float64[])
end

function jth_hess_residual(nlp :: FeasibilityFormNLS, x :: AbstractVector, i :: Int)
  increment!(nlp, :neval_jhess_residual)
  n = nlp.meta.nvar
  return spzeros(n, n)
end

function hprod_residual!(nlp :: FeasibilityFormNLS, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  increment!(nlp, :neval_hprod_residual)
  fill!(Hiv, 0.0)
  return Hiv
end
