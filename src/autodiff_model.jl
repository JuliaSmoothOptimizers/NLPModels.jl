using ForwardDiff

export ADNLPModel

mutable struct ADNLPModel <: AbstractNLPModel
  meta :: NLPModelMeta

  counters :: Counters

  # Functions
  f
  c
end

show_header(io :: IO, nlp :: ADNLPModel) = println(io, "ADNLPModel - Model with automatic differentiation")

"""
    ADNLPModel(f, x0)
    ADNLPModel(f, x0, lvar, uvar)
    ADNLPModel(f, x0, c, lcon, ucon)
    ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon)

ADNLPModel is an AbstractNLPModel using ForwardDiff to compute the derivatives.
The problem is defined as

     min  f(x)
    s.to  lcon ≤ c(x) ≤ ucon
          lvar ≤   x  ≤ uvar.

The following keyword arguments are available to all constructors:

- `name`: The name of the model (default: "Generic")

The following keyword arguments are available to the constructors for constrained problems:

- `lin`: An array of indexes of the linear constraints (default: `Int[]`)
- `y0`: An inital estimate to the Lagrangian multipliers (default: zeros)
"""
function ADNLPModel end

function ADNLPModel(f, x0::AbstractVector{T}; name::String = "Generic") where T
  nvar = length(x0)
  @lencheck nvar x0

  nnzh = nvar * (nvar + 1) / 2

  meta = NLPModelMeta(nvar, x0=x0, nnzh=nnzh, minimize=true, islp=false, name=name)

  return ADNLPModel(meta, Counters(), f, x->T[])
end

function ADNLPModel(f, x0::AbstractVector{T}, lvar::AbstractVector, uvar::AbstractVector;
                    name::String = "Generic") where T
  nvar = length(x0)
  @lencheck nvar x0 lvar uvar

  nnzh = nvar * (nvar + 1) / 2

  meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, nnzh=nnzh, minimize=true, islp=false, name=name)

  return ADNLPModel(meta, Counters(), f, x->T[])
end

function ADNLPModel(f, x0::AbstractVector{T}, c, lcon::AbstractVector, ucon::AbstractVector;
                    y0::AbstractVector=fill!(similar(lcon), zero(T)),
                    name::String = "Generic", lin::AbstractVector{<: Integer}=Int[]) where T
  
  nvar = length(x0)
  ncon = length(lcon)
  @lencheck nvar x0
  @lencheck ncon ucon y0

  nnzh = nvar * (nvar + 1) / 2
  nnzj = nvar * ncon

  nln = setdiff(1:ncon, lin)

  meta = NLPModelMeta(nvar, x0=x0, ncon=ncon, y0=y0, lcon=lcon, ucon=ucon,
    nnzj=nnzj, nnzh=nnzh, lin=lin, nln=nln, minimize=true, islp=false, name=name)

  return ADNLPModel(meta, Counters(), f, c)
end

function ADNLPModel(f, x0::AbstractVector{T}, lvar::AbstractVector, uvar::AbstractVector,
                    c, lcon::AbstractVector, ucon::AbstractVector;
                    y0::AbstractVector=fill!(similar(lcon), zero(T)),
                    name::String = "Generic", lin::AbstractVector{<: Integer}=Int[]) where T

  nvar = length(x0)
  ncon = length(lcon)
  @lencheck nvar x0 lvar uvar
  @lencheck ncon y0 ucon

  nnzh = nvar * (nvar + 1) / 2
  nnzj = nvar * ncon

  nln = setdiff(1:ncon, lin)

  meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, y0=y0,
    lcon=lcon, ucon=ucon, nnzj=nnzj, nnzh=nnzh, lin=lin, nln=nln, minimize=true,
    islp=false, name=name)

  return ADNLPModel(meta, Counters(), f, c)
end

function obj(nlp :: ADNLPModel, x :: AbstractVector)
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_obj)
  return nlp.f(x)
end

function grad!(nlp :: ADNLPModel, x :: AbstractVector, g :: AbstractVector)
  @lencheck nlp.meta.nvar x g
  increment!(nlp, :neval_grad)
  ForwardDiff.gradient!(g, nlp.f, x)
  return g
end

function cons!(nlp :: ADNLPModel, x :: AbstractVector, c :: AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon c
  increment!(nlp, :neval_cons)
  c .= nlp.c(x)
  return c
end

function jac(nlp :: ADNLPModel, x :: AbstractVector)
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_jac)
  return ForwardDiff.jacobian(nlp.c, x)
end

function jac_structure!(nlp :: ADNLPModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck nlp.meta.nnzj rows cols
  m, n = nlp.meta.ncon, nlp.meta.nvar
  I = ((i,j) for i = 1:m, j = 1:n)
  rows .= getindex.(I, 1)[:]
  cols .= getindex.(I, 2)[:]
  return rows, cols
end

function jac_coord!(nlp :: ADNLPModel, x :: AbstractVector, vals :: AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzj vals
  increment!(nlp, :neval_jac)
  Jx = ForwardDiff.jacobian(nlp.c, x)
  vals .= Jx[:]
  return vals
end

function jprod!(nlp :: ADNLPModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck nlp.meta.nvar x v
  @lencheck nlp.meta.ncon Jv
  increment!(nlp, :neval_jprod)
  Jv .= ForwardDiff.derivative(t -> nlp.c(x + t * v), 0)
  return Jv
end

function jtprod!(nlp :: ADNLPModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.ncon v
  increment!(nlp, :neval_jtprod)
  Jtv .= ForwardDiff.gradient(x -> dot(nlp.c(x), v), x)
  return Jtv
end

function hess(nlp :: ADNLPModel, x :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_hess)
  ℓ(x) = obj_weight * nlp.f(x)
  Hx = ForwardDiff.hessian(ℓ, x)
  return tril(Hx)
end

function hess(nlp :: ADNLPModel, x :: AbstractVector, y :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon y
  increment!(nlp, :neval_hess)
  ℓ(x) = obj_weight * nlp.f(x) + dot(nlp.c(x), y)
  Hx = ForwardDiff.hessian(ℓ, x)
  return tril(Hx)
end

function hess_structure!(nlp :: ADNLPModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  n = nlp.meta.nvar
  @lencheck nlp.meta.nnzh rows cols
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows .= getindex.(I, 1)
  cols .= getindex.(I, 2)
  return rows, cols
end

function hess_coord!(nlp :: ADNLPModel, x :: AbstractVector, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzh vals
  increment!(nlp, :neval_hess)
  ℓ(x) = obj_weight * nlp.f(x)
  Hx = ForwardDiff.hessian(ℓ, x)
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function hess_coord!(nlp :: ADNLPModel, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon y
  @lencheck nlp.meta.nnzh vals
  increment!(nlp, :neval_hess)
  ℓ(x) = obj_weight * nlp.f(x) + dot(nlp.c(x), y)
  Hx = ForwardDiff.hessian(ℓ, x)
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function hprod!(nlp :: ADNLPModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  n = nlp.meta.nvar
  @lencheck n x v Hv
  increment!(nlp, :neval_hprod)
  ℓ(x) = obj_weight * nlp.f(x)
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
  return Hv
end

function hprod!(nlp :: ADNLPModel, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  n = nlp.meta.nvar
  @lencheck n x v Hv
  @lencheck nlp.meta.ncon y
  increment!(nlp, :neval_hprod)
  ℓ(x) = obj_weight * nlp.f(x) + dot(nlp.c(x), y)
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
  return Hv
end

function ghjvprod!(nlp :: ADNLPModel, x :: AbstractVector, g :: AbstractVector, v :: AbstractVector, gHv :: AbstractVector) 
  @lencheck nlp.meta.nvar x g v
  @lencheck nlp.meta.ncon gHv
  increment!(nlp, :neval_hprod)
  gHv .= ForwardDiff.derivative(t -> ForwardDiff.derivative(s -> nlp.c(x + s * g + t * v), 0), 0)
  return gHv
end
