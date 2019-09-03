using ForwardDiff

export ADNLSModel

"""ADNLSModel is an Nonlinear Least Squares model using ForwardDiff to
compute the derivatives.

````
ADNLSModel(F, x0, m; lvar = [-∞,…,-∞], uvar = [∞,…,∞], y0 = zeros,
  c = NotImplemented, lcon = [-∞,…,-∞], ucon = [∞,…,∞], name = "Generic")
````

  - `F` - The residual function \$F\$. Should be callable;
  - `x0 :: AbstractVector` - The initial point of the problem;
  - `m :: Int` - The dimension of \$F(x)\$, i.e., the number of
  equations in the nonlinear system.

The other parameters are as in `ADNLPModel`.
"""
mutable struct ADNLSModel <: AbstractNLSModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  counters :: NLSCounters

  # Function
  F
  c
end

function ADNLSModel(F, x0 :: AbstractVector, m :: Int;
                    name :: String = "Generic",
                    lvar :: AbstractVector = fill(-eltype(x0)(Inf), length(x0)),
                    uvar :: AbstractVector = fill( eltype(x0)(Inf), length(x0)),
                    c = (args...)->throw(NotImplementedError("cons")),
                    lcon :: AbstractVector = eltype(x0)[],
                    ucon :: AbstractVector = eltype(x0)[],
                    y0 :: AbstractVector = zeros(eltype(x0), max(length(lcon), length(ucon)))
                   )
  nvar = length(x0)
  ncon = maximum([length(lcon); length(ucon); length(y0)])
  if !(length(lcon) == length(ucon) == length(y0))
    error("lcon, ucon and y0 need to be the same length")
  end
  nnzj = nvar * ncon

  meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, y0=y0,
                      lcon=lcon, ucon=ucon, nnzj=nnzj, name=name)
  nls_meta = NLSMeta(m, nvar, nnzj=m * nvar, nnzh=div(nvar * (nvar + 1), 2))

  return ADNLSModel(meta, nls_meta, NLSCounters(), F, c)
end

ADNLSModel(F, n :: Int, m :: Int; kwargs...) = ADNLSModel(F, zeros(n), m; kwargs...)

function residual!(nls :: ADNLSModel, x :: AbstractVector, Fx :: AbstractVector)
  increment!(nls, :neval_residual)
  Fx[1:nls.nls_meta.nequ] = nls.F(x)
  return Fx
end

function jac_residual(nls :: ADNLSModel, x :: AbstractVector)
  increment!(nls, :neval_jac_residual)
  return ForwardDiff.jacobian(nls.F, x)
end

function jac_structure_residual!(nls :: ADNLSModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  m, n = nls.nls_meta.nequ, nls.meta.nvar
  I = ((i,j) for i = 1:m, j = 1:n)
  rows .= getindex.(I, 1)[:]
  cols .= getindex.(I, 2)[:]
  return rows, cols
end

function jac_coord_residual!(nls :: ADNLSModel, x :: AbstractVector, vals :: AbstractVector)
  increment!(nls, :neval_jac_residual)
  Jx = ForwardDiff.jacobian(nls.F, x)
  vals .= Jx[:]
  return vals
end

function jprod_residual!(nls :: ADNLSModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nls, :neval_jprod_residual)
  Jv[1:nls.nls_meta.nequ] = ForwardDiff.jacobian(nls.F, x) * v
  return Jv
end

function jtprod_residual!(nls :: ADNLSModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nls, :neval_jtprod_residual)
  Jtv[1:nls.meta.nvar] = ForwardDiff.jacobian(nls.F, x)' * v
  return Jtv
end

function hess_residual(nls :: ADNLSModel, x :: AbstractVector, v :: AbstractVector)
  increment!(nls, :neval_hess_residual)
  return tril(ForwardDiff.jacobian(x -> ForwardDiff.jacobian(nls.F, x)' * v, x))
end

function hess_structure_residual!(nls :: ADNLSModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  n = nls.meta.nvar
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows .= getindex.(I, 1)
  cols .= getindex.(I, 2)
  return rows, cols
end

function hess_coord_residual!(nls :: ADNLSModel, x :: AbstractVector, v :: AbstractVector, vals :: AbstractVector)
  increment!(nls, :neval_hess_residual)
  Hx = ForwardDiff.jacobian(x->ForwardDiff.jacobian(nls.F, x)' * v, x)
  k = 1
  for j = 1:nls.meta.nvar
    for i = j:nls.meta.nvar
      vals[k] = Hx[i,j]
      k += 1
    end
  end
  return vals
end

function jth_hess_residual(nls :: ADNLSModel, x :: AbstractVector, i :: Int)
  increment!(nls, :neval_jhess_residual)
  return tril(ForwardDiff.hessian(x->nls.F(x)[i], x))
end

function hprod_residual!(nls :: ADNLSModel, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  increment!(nls, :neval_hprod_residual)
  Hiv[1:nls.meta.nvar] = ForwardDiff.hessian(x->nls.F(x)[i], x) * v
  return Hiv
end

function cons!(nls :: ADNLSModel, x :: AbstractVector, c :: AbstractVector)
  increment!(nls, :neval_cons)
  c[1:nls.meta.ncon] = nls.c(x)
  return c
end

function jac(nls :: ADNLSModel, x :: AbstractVector)
  increment!(nls, :neval_jac)
  return ForwardDiff.jacobian(nls.c, x)
end

function jac_structure!(nls :: ADNLSModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  m, n = nls.meta.ncon, nls.meta.nvar
  I = ((i,j) for i = 1:m, j = 1:n)
  rows .= getindex.(I, 1)[:]
  cols .= getindex.(I, 2)[:]
  return rows, cols
end

function jac_coord!(nls :: ADNLSModel, x :: AbstractVector, vals :: AbstractVector)
  Jx = ForwardDiff.jacobian(nls.c, x)
  vals .= Jx[:]
  return vals
end

function jprod!(nls :: ADNLSModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nls, :neval_jprod)
  Jv[1:nls.meta.ncon] = ForwardDiff.jacobian(nls.c, x) * v
  return Jv
end

function jtprod!(nls :: ADNLSModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nls, :neval_jtprod)
  Jtv[1:nls.meta.nvar] = ForwardDiff.jacobian(nls.c, x)' * v
  return Jtv
end

function hess(nls :: ADNLSModel, x :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  increment!(nls, :neval_hess)
  ℓ(x) = obj_weight * sum(nls.F(x).^2) / 2
  Hx = ForwardDiff.hessian(ℓ, x)
  return tril(Hx)
end

function hess(nls :: ADNLSModel, x :: AbstractVector, y :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  increment!(nls, :neval_hess)
  ℓ(x) = obj_weight * sum(nls.F(x).^2) / 2 + dot(y, nls.c(x))
  Hx = ForwardDiff.hessian(ℓ, x)
  return tril(Hx)
end

function hess_structure!(nls :: ADNLSModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  n = nls.meta.nvar
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows .= getindex.(I, 1)
  cols .= getindex.(I, 2)
  return rows, cols
end

function hess_coord!(nls :: ADNLSModel, x :: AbstractVector, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  increment!(nls, :neval_hess)
  ℓ(x) = obj_weight * sum(nls.F(x).^2) / 2
  Hx = ForwardDiff.hessian(ℓ, x)
  k = 1
  for j = 1:nls.meta.nvar
    for i = j:nls.meta.nvar
      vals[k] = Hx[i,j]
      k += 1
    end
  end
  return vals
end

function hess_coord!(nls :: ADNLSModel, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  increment!(nls, :neval_hess)
  ℓ(x) = obj_weight * sum(nls.F(x).^2) / 2 + dot(y, nls.c(x))
  Hx = ForwardDiff.hessian(ℓ, x)
  k = 1
  for j = 1:nls.meta.nvar
    for i = j:nls.meta.nvar
      vals[k] = Hx[i,j]
      k += 1
    end
  end
  return vals
end

function hprod!(nls :: ADNLSModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight = one(eltype(x)))
  increment!(nls, :neval_hprod)
  ℓ(x) = obj_weight * sum(nls.F(x).^2) / 2
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
  return Hv
end

function hprod!(nls :: ADNLSModel, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector;
                obj_weight = one(eltype(x)))
  increment!(nls, :neval_hprod)
  ℓ(x) = obj_weight * sum(nls.F(x).^2) / 2 + dot(y, nls.c(x))
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
  return Hv
end
