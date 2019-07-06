using ForwardDiff

export ADNLPModel, obj, grad, grad!, cons, cons!, jac, jprod,
       jprod!, jtprod, jtprod!, hess, hprod, hprod!

@doc raw"""ADNLPModel is an AbstractNLPModel using ForwardDiff to compute the
derivatives.
In this interface, the objective function ``f`` and an initial estimate are
required. If there are constraints, the function
``c:ℝⁿ → ℝᵐ``  and the vectors
``c_L`` and ``c_U`` also need to be passed. Bounds on the variables and an
inital estimate to the Lagrangian multipliers can also be provided.

````
ADNLPModel(f, x0; lvar = [-∞,…,-∞], uvar = [∞,…,∞], y0 = zeros,
  c = NotImplemented, lcon = [-∞,…,-∞], ucon = [∞,…,∞], name = "Generic")
````

  - `f :: Function` - The objective function ``f``;
  - `x0 :: AbstractVector` - The initial point of the problem;
  - `lvar :: AbstractVector` - ``ℓ``, the lower bound of the variables;
  - `uvar :: AbstractVector` - ``u``, the upper bound of the variables;
  - `c :: Function` - The constraints function ``c``;
  - `y0 :: AbstractVector` - The initial value of the Lagrangian estimates;
  - `lcon :: AbstractVector` - ``c_L``, the lower bounds of the constraints function;
  - `ucon :: AbstractVector` - ``c_U``, the upper bounds of the constraints function;
  - `name :: String` - A name for the model.

The functions follow the same restrictions of ForwardDiff functions, summarised
here:

  - The function can only be composed of generic Julia functions;
  - The function must accept only one argument;
  - The function's argument must accept a subtype of AbstractVector;
  - The function should be type-stable.

For contrained problems, the function ``c`` is required, and it must return
an array even when m = 1,
and ``c_L`` and ``c_U`` should be passed, otherwise the problem is ill-formed.
For equality constraints, the corresponding index of ``c_L`` and ``c_U`` should be the
same.
"""
mutable struct ADNLPModel <: AbstractNLPModel
  meta :: NLPModelMeta

  counters :: Counters

  # Functions
  f :: Function
  c :: Function
end

function ADNLPModel(f::Function, x0::AbstractVector; y0::AbstractVector = eltype(x0)[],
                    lvar::AbstractVector = eltype(x0)[], uvar::AbstractVector = eltype(x0)[],
                    lcon::AbstractVector = eltype(x0)[], ucon::AbstractVector = eltype(x0)[],
    c::Function = (args...)->throw(NotImplementedError("cons")),
    name::String = "Generic", lin::AbstractVector{<: Integer}=Int[])

  nvar = length(x0)
  length(lvar) == 0 && (lvar = -Inf*ones(nvar))
  length(uvar) == 0 && (uvar =  Inf*ones(nvar))
  ncon = maximum([length(lcon); length(ucon); length(y0)])

  A = ForwardDiff.hessian(f, x0)
  for i = 1:ncon
    A += ForwardDiff.hessian(x->c(x)[i], x0) * (-1)^i
  end
  nnzh = nvar * (nvar + 1) / 2
  nnzj = 0

  if ncon > 0
    length(lcon) == 0 && (lcon = -Inf*ones(ncon))
    length(ucon) == 0 && (ucon =  Inf*ones(ncon))
    length(y0) == 0   && (y0 = zeros(ncon))
    nnzj = nvar * ncon
  end
  nln = setdiff(1:ncon, lin)

  meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, y0=y0,
    lcon=lcon, ucon=ucon, nnzj=nnzj, nnzh=nnzh, lin=lin, nln=nln, minimize=true,
    islp=false, name=name)

  return ADNLPModel(meta, Counters(), f, c)
end

function obj(nlp :: ADNLPModel, x :: AbstractVector)
  increment!(nlp, :neval_obj)
  return nlp.f(x)
end

function grad!(nlp :: ADNLPModel, x :: AbstractVector, g :: AbstractVector)
  increment!(nlp, :neval_grad)
  ForwardDiff.gradient!(view(g, 1:length(x)), nlp.f, x)
  return g
end

function cons!(nlp :: ADNLPModel, x :: AbstractVector, c :: AbstractVector)
  increment!(nlp, :neval_cons)
  c[1:nlp.meta.ncon] = nlp.c(x)
  return c
end

function jac(nlp :: ADNLPModel, x :: AbstractVector)
  increment!(nlp, :neval_jac)
  return ForwardDiff.jacobian(nlp.c, x)
end

function jac_structure!(nlp :: ADNLPModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  m, n = nlp.meta.ncon, nlp.meta.nvar
  I = ((i,j) for i = 1:m, j = 1:n)
  rows[1 : nlp.meta.nnzj] .= getindex.(I, 1)[:]
  cols[1 : nlp.meta.nnzj] .= getindex.(I, 2)[:]
  return rows, cols
end

function jac_coord!(nlp :: ADNLPModel, x :: AbstractVector, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer}, vals :: AbstractVector)
  Jx = jac(nlp, x)
  vals[1 : nlp.meta.nnzj] .= Jx[:]
  return rows, cols, vals
end

function jprod!(nlp :: ADNLPModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jprod)
  Jv[1:nlp.meta.ncon] = ForwardDiff.jacobian(nlp.c, x) * v
  return Jv
end

function jtprod!(nlp :: ADNLPModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod)
  Jtv[1:nlp.meta.nvar] = ForwardDiff.jacobian(nlp.c, x)' * v
  return Jtv
end

function hess(nlp :: ADNLPModel, x :: AbstractVector; obj_weight :: Real = one(eltype(x)), y :: AbstractVector = eltype(x)[])
  increment!(nlp, :neval_hess)
  Hx = obj_weight == 0.0 ? zeros(nlp.meta.nvar, nlp.meta.nvar) :
       ForwardDiff.hessian(nlp.f, x) * obj_weight
  for i = 1:min(length(y), nlp.meta.ncon)
    if y[i] != 0.0
      Hx += ForwardDiff.hessian(x->nlp.c(x)[i], x) * y[i]
    end
  end
  return tril(Hx)
end

function hess_structure!(nlp :: ADNLPModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  n = nlp.meta.nvar
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows[1 : nlp.meta.nnzh] .= getindex.(I, 1)
  cols[1 : nlp.meta.nnzh] .= getindex.(I, 2)
  return rows, cols
end

function hess_coord!(nlp :: ADNLPModel, x :: AbstractVector, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer}, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)), y :: AbstractVector = eltype(x)[])
  Hx = hess(nlp, x, obj_weight=obj_weight, y=y)
  for k = 1:nlp.meta.nnzh
    i, j = rows[k], cols[k]
    vals[k] = Hx[i,j]
  end
  return rows, cols, vals
end

function hprod!(nlp :: ADNLPModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector;
                obj_weight :: Real = one(eltype(x)), y :: AbstractVector = eltype(x)[])
  increment!(nlp, :neval_hprod)
  n = nlp.meta.nvar
  Hv[1:n] = obj_weight == 0.0 ? zeros(nlp.meta.nvar) :
          ForwardDiff.hessian(nlp.f, x) * v * obj_weight
  for i = 1:min(length(y), nlp.meta.ncon)
    if y[i] != 0.0
      Hv[1:n] += ForwardDiff.hessian(x->nlp.c(x)[i], x) * v * y[i]
    end
  end
  return Hv
end
