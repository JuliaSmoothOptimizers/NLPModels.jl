using ForwardDiff

export ADNLSModel

mutable struct ADNLSModel <: AbstractNLSModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  counters :: NLSCounters

  # Function
  F
  c
end

show_header(io :: IO, nls :: ADNLSModel) = println(io, "ADNLSModel - Nonlinear least-squares model with automatic differentiation")

"""
    ADNLSModel(F, x0, nequ)
    ADNLSModel(F, x0, nequ, lvar, uvar)
    ADNLSModel(F, x0, nequ, c, lcon, ucon)
    ADNLSModel(F, x0, nequ, lvar, uvar, c, lcon, ucon)

ADNLSModel is an Nonlinear Least Squares model using ForwardDiff to
compute the derivatives.
The problem is defined as

     min  ½‖F(x)‖²
    s.to  lcon ≤ c(x) ≤ ucon
          lvar ≤   x  ≤ uvar

The following keyword arguments are available to all constructors:

- `linequ`: An array of indexes of the linear equations (default: `Int[]`)
- `name`: The name of the model (default: "Generic")

The following keyword arguments are available to the constructors for constrained problems:

- `lin`: An array of indexes of the linear constraints (default: `Int[]`)
- `y0`: An inital estimate to the Lagrangian multipliers (default: zeros)
"""
function ADNLSModel end

function ADNLSModel(F, x0 :: AbstractVector{T}, nequ :: Integer;
                    linequ :: AbstractVector{<: Integer} = Int[],
                    name :: String = "Generic",
                   ) where T

  nvar = length(x0)

  meta = NLPModelMeta(nvar, x0=x0, name=name)
  nlnequ = setdiff(1:nequ, linequ)
  nls_meta = NLSMeta(nequ, nvar, nnzj=nequ * nvar, nnzh=div(nvar * (nvar + 1), 2), lin=linequ, nln=nlnequ)

  return ADNLSModel(meta, nls_meta, NLSCounters(), F, x->T[])
end

function ADNLSModel(F, x0 :: AbstractVector{T}, nequ :: Integer,
                    lvar :: AbstractVector, uvar :: AbstractVector;
                    linequ :: AbstractVector{<: Integer} = Int[],
                    name :: String = "Generic",
                   ) where T

  nvar = length(x0)
  @lencheck nvar lvar uvar

  meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, name=name)
  nlnequ = setdiff(1:nequ, linequ)
  nls_meta = NLSMeta(nequ, nvar, nnzj=nequ * nvar, nnzh=div(nvar * (nvar + 1), 2), lin=linequ, nln=nlnequ)

  return ADNLSModel(meta, nls_meta, NLSCounters(), F, x->T[])
end

function ADNLSModel(F, x0 :: AbstractVector{T}, nequ :: Integer,
                    c, lcon :: AbstractVector, ucon :: AbstractVector;
                    y0 :: AbstractVector = fill!(similar(lcon), zero(T)),
                    lin :: AbstractVector{<: Integer} = Int[],
                    linequ :: AbstractVector{<: Integer} = Int[],
                    name :: String = "Generic",
                   ) where T
  
  nvar = length(x0)
  ncon = length(lcon)
  @lencheck ncon ucon y0
  nnzj = nvar * ncon

  nln = setdiff(1:ncon, lin)
  meta = NLPModelMeta(nvar, x0=x0, ncon=ncon, y0=y0, lcon=lcon, ucon=ucon,
                      nnzj=nnzj, name=name, lin=lin, nln=nln)
  nlnequ = setdiff(1:nequ, linequ)
  nls_meta = NLSMeta(nequ, nvar, nnzj=nequ * nvar, nnzh=div(nvar * (nvar + 1), 2), lin=linequ, nln=nlnequ)

  return ADNLSModel(meta, nls_meta, NLSCounters(), F, c)
end

function ADNLSModel(F, x0 :: AbstractVector{T}, nequ :: Integer,
                    lvar :: AbstractVector, uvar :: AbstractVector,
                    c, lcon :: AbstractVector, ucon :: AbstractVector;
                    y0 :: AbstractVector = fill!(similar(lcon), zero(T)),
                    lin :: AbstractVector{<: Integer} = Int[],
                    linequ :: AbstractVector{<: Integer} = Int[],
                    name :: String = "Generic",
                   ) where T
  
  nvar = length(x0)
  ncon = length(lcon)
  @lencheck nvar lvar uvar
  @lencheck ncon ucon y0
  nnzj = nvar * ncon

  nln = setdiff(1:ncon, lin)
  meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, y0=y0,
                      lcon=lcon, ucon=ucon, nnzj=nnzj, name=name, lin=lin, nln=nln)
  nlnequ = setdiff(1:nequ, linequ)
  nls_meta = NLSMeta(nequ, nvar, nnzj=nequ * nvar, nnzh=div(nvar * (nvar + 1), 2), lin=linequ, nln=nlnequ)

  return ADNLSModel(meta, nls_meta, NLSCounters(), F, c)
end

function residual!(nls :: ADNLSModel, x :: AbstractVector, Fx :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ Fx
  increment!(nls, :neval_residual)
  Fx .= nls.F(x)
  return Fx
end

function jac_residual(nls :: ADNLSModel, x :: AbstractVector)
  @lencheck nls.meta.nvar x
  increment!(nls, :neval_jac_residual)
  return ForwardDiff.jacobian(nls.F, x)
end

function jac_structure_residual!(nls :: ADNLSModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck nls.nls_meta.nnzj rows cols
  m, n = nls.nls_meta.nequ, nls.meta.nvar
  I = ((i,j) for i = 1:m, j = 1:n)
  rows .= getindex.(I, 1)[:]
  cols .= getindex.(I, 2)[:]
  return rows, cols
end

function jac_coord_residual!(nls :: ADNLSModel, x :: AbstractVector, vals :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nnzj vals
  increment!(nls, :neval_jac_residual)
  Jx = ForwardDiff.jacobian(nls.F, x)
  vals .= Jx[:]
  return vals
end

function jprod_residual!(nls :: ADNLSModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck nls.meta.nvar x v
  @lencheck nls.nls_meta.nequ Jv
  increment!(nls, :neval_jprod_residual)
  Jv .= ForwardDiff.derivative(t -> nls.F(x + t * v), 0)
  return Jv
end

function jtprod_residual!(nls :: ADNLSModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck nls.meta.nvar x Jtv
  @lencheck nls.nls_meta.nequ v
  increment!(nls, :neval_jtprod_residual)
  Jtv .= ForwardDiff.gradient(x -> dot(nls.F(x), v), x)
  return Jtv
end

function hess_residual(nls :: ADNLSModel, x :: AbstractVector, v :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ v
  increment!(nls, :neval_hess_residual)
  return tril(ForwardDiff.jacobian(x -> ForwardDiff.jacobian(nls.F, x)' * v, x))
end

function hess_structure_residual!(nls :: ADNLSModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck nls.nls_meta.nnzh rows cols
  n = nls.meta.nvar
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows .= getindex.(I, 1)
  cols .= getindex.(I, 2)
  return rows, cols
end

function hess_coord_residual!(nls :: ADNLSModel, x :: AbstractVector, v :: AbstractVector, vals :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ v
  @lencheck nls.nls_meta.nnzh vals
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
  @lencheck nls.meta.nvar x
  increment!(nls, :neval_jhess_residual)
  return tril(ForwardDiff.hessian(x->nls.F(x)[i], x))
end

function hprod_residual!(nls :: ADNLSModel, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  @lencheck nls.meta.nvar x v Hiv
  increment!(nls, :neval_hprod_residual)
  Hiv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(x -> nls.F(x)[i], x + t * v), 0)
  return Hiv
end

function cons!(nls :: ADNLSModel, x :: AbstractVector, c :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.meta.ncon c
  increment!(nls, :neval_cons)
  c .= nls.c(x)
  return c
end

function jac(nls :: ADNLSModel, x :: AbstractVector)
  @lencheck nls.meta.nvar x
  increment!(nls, :neval_jac)
  return ForwardDiff.jacobian(nls.c, x)
end

function jac_structure!(nls :: ADNLSModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck nls.meta.nnzj rows cols
  m, n = nls.meta.ncon, nls.meta.nvar
  I = ((i,j) for i = 1:m, j = 1:n)
  rows .= getindex.(I, 1)[:]
  cols .= getindex.(I, 2)[:]
  return rows, cols
end

function jac_coord!(nls :: ADNLSModel, x :: AbstractVector, vals :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.meta.nnzj vals
  Jx = ForwardDiff.jacobian(nls.c, x)
  vals .= Jx[:]
  return vals
end

function jprod!(nls :: ADNLSModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck nls.meta.nvar x v
  @lencheck nls.meta.ncon Jv
  increment!(nls, :neval_jprod)
  Jv .= ForwardDiff.derivative(t -> nls.c(x + t * v), 0)
  return Jv
end

function jtprod!(nls :: ADNLSModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck nls.meta.nvar x Jtv
  @lencheck nls.meta.ncon v
  increment!(nls, :neval_jtprod)
  Jtv .= ForwardDiff.gradient(x -> dot(nls.c(x), v), x)
  return Jtv
end

function hess(nls :: ADNLSModel, x :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  @lencheck nls.meta.nvar x
  increment!(nls, :neval_hess)
  ℓ(x) = obj_weight * sum(nls.F(x).^2) / 2
  Hx = ForwardDiff.hessian(ℓ, x)
  return tril(Hx)
end

function hess(nls :: ADNLSModel, x :: AbstractVector, y :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  @lencheck nls.meta.nvar x
  @lencheck nls.meta.ncon y
  increment!(nls, :neval_hess)
  ℓ(x) = obj_weight * sum(nls.F(x).^2) / 2 + dot(y, nls.c(x))
  Hx = ForwardDiff.hessian(ℓ, x)
  return tril(Hx)
end

function hess_structure!(nls :: ADNLSModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck nls.meta.nnzh rows cols
  n = nls.meta.nvar
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows .= getindex.(I, 1)
  cols .= getindex.(I, 2)
  return rows, cols
end

function hess_coord!(nls :: ADNLSModel, x :: AbstractVector, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  @lencheck nls.meta.nvar x
  @lencheck nls.meta.nnzh vals
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
  @lencheck nls.meta.nvar x
  @lencheck nls.meta.ncon y
  @lencheck nls.meta.nnzh vals
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
  @lencheck nls.meta.nvar x v Hv
  increment!(nls, :neval_hprod)
  ℓ(x) = obj_weight * sum(nls.F(x).^2) / 2
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
  return Hv
end

function hprod!(nls :: ADNLSModel, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector;
                obj_weight = one(eltype(x)))
  @lencheck nls.meta.nvar x v Hv
  @lencheck nls.meta.ncon y
  increment!(nls, :neval_hprod)
  ℓ(x) = obj_weight * sum(nls.F(x).^2) / 2 + dot(y, nls.c(x))
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
  return Hv
end
