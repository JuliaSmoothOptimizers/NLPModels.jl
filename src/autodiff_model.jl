using ForwardDiff

export ADNLPModel, obj, grad, grad!, cons, cons!, jac_coord, jac, jprod,
       jprod!, jtprod, jtprod!, hess, hprod, hprod!

"""ADNLPModel is an AbstractNLPModel using ForwardDiff to computer the
derivatives.
In this interface, the objective function \$f\$ and an initial estimate are
required. If there are constraints, the function
\$c:\\mathbb{R}^n\\rightarrow\\mathbb{R}^m\$  and the vectors
\$c_L\$ and \$c_U\$ also need to be passed. Bounds on the variables and an
inital estimate to the Lagrangian multipliers can also be provided.

````
ADNLPModel(f, x0; lvar = [-∞,…,-∞], uvar = [∞,…,∞], y0=zeros,
  c = NotImplemented, lcon = [-∞,…,-∞], ucon = [∞,…,∞], name = "Generic")
````

  - `f :: Function` - The objective function \$f\$;
  - `x0 :: Vector` - The initial point of the problem;
  - `lvar :: Vector` - \$\\ell\$, the lower bound of the variables;
  - `uvar :: Vector` - \$u\$, the upper bound of the variables;
  - `c :: Function` - The constraints function \$c\$;
  - `y0 :: Vector` - The initial value of the Lagrangian estimates;
  - `lcon :: Vector` - \$c_L\$, the lower bounds of the constraints function;
  - `ucon :: Vector` - \$c_U\$, the upper bounds of the constraints function;
  - `name :: AbstractString` - A name for the model.

The functions follow the same restrictions of ForwardDiff functions, summarised
here:

  - The function can only be composed of generic Julia functions;
  - The function must accept only one argument;
  - The function's argument must accept a subtype of Vector;
  - The function should be type-stable.

For contrained problems, the function \$c\$ is required, and it must return
an array even when m = 1,
and \$c_L\$ and \$c_U\$ should be passed, otherwise the problem is ill-formed.
For equality constraints, the corresponding index of \$c_L\$ and \$c_U\$ should be the
same.
"""
type ADNLPModel <: AbstractNLPModel
  meta :: NLPModelMeta

  counters :: Counters

  # Functions
  f :: Function
  c :: Function
end

function ADNLPModel(f::Function, x0::Vector; y0::Vector = [],
    lvar::Vector = [], uvar::Vector = [], lcon::Vector = [], ucon::Vector = [],
    c::Function = (args...)->throw(NotImplementedError("cons")),
    name::AbstractString = "Generic")

  nvar = length(x0)
  length(lvar) == 0 && (lvar = -Inf*ones(nvar))
  length(uvar) == 0 && (uvar =  Inf*ones(nvar))
  ncon = maximum([length(lcon); length(ucon); length(y0)])

  A = ForwardDiff.hessian(f, x0)
  for i = 1:ncon
    A += ForwardDiff.hessian(x->c(x)[i], x0) * (-1)^i
  end
  nnzh = typeof(A) <: SparseMatrixCSC ? nnz(A) : length(A)
  nnzj = 0

  if ncon > 0
    length(lcon) == 0 && (lcon = -Inf*ones(ncon))
    length(ucon) == 0 && (ucon =  Inf*ones(ncon))
    length(y0) == 0   && (y0 = zeros(ncon))
    A = ForwardDiff.jacobian(c, x0)
    nnzj = typeof(A) <: SparseMatrixCSC ? nnz(A) : length(A)
  end
  lin = []
  nln = collect(1:ncon)

  meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, y0=y0,
    lcon=lcon, ucon=ucon, nnzj=nnzj, nnzh=nnzh, lin=lin, nln=nln, minimize=true,
    islp=false, name=name)

  return ADNLPModel(meta, Counters(), f, c)
end

function obj(nlp :: ADNLPModel, x :: Vector)
  nlp.counters.neval_obj += 1
  return nlp.f(x)
end

function grad(nlp :: ADNLPModel, x :: Vector)
  nlp.counters.neval_grad += 1
  return ForwardDiff.gradient(nlp.f, x)
end

function grad!(nlp :: ADNLPModel, x :: Vector, g :: Vector)
  nlp.counters.neval_grad += 1
  ForwardDiff.gradient!(sub(g, 1:length(x)), nlp.f, x)
end

function cons(nlp :: ADNLPModel, x :: Vector)
  nlp.counters.neval_cons += 1
  return nlp.c(x)
end

function cons!(nlp :: ADNLPModel, x :: Vector, c :: Vector)
  nlp.counters.neval_cons += 1
  c[:] = nlp.c(x)
  return c
end

function jac_coord(nlp :: ADNLPModel, x :: Vector)
  nlp.counters.neval_jac += 1
  J = ForwardDiff.jacobian(nlp.c, x)
  return typeof(J) <: Matrix ? findnz(sparse(J)) : findnz(J)
end

function jac(nlp :: ADNLPModel, x :: Vector)
  nlp.counters.neval_jac += 1
  return ForwardDiff.jacobian(nlp.c, x)
end

function jprod(nlp :: ADNLPModel, x :: Vector, v :: Vector)
  nlp.counters.neval_jprod += 1
  return ForwardDiff.jacobian(nlp.c, x) * v
end

function jprod!(nlp :: ADNLPModel, x :: Vector, v :: Vector, Jv :: Vector)
  nlp.counters.neval_jprod += 1
  Jv[1:nlp.meta.ncon] = ForwardDiff.jacobian(nlp.c, x) * v
  return Jv
end

function jtprod(nlp :: ADNLPModel, x :: Vector, v :: Vector)
  nlp.counters.neval_jtprod += 1
  return ForwardDiff.jacobian(nlp.c, x)' * v
end

function jtprod!(nlp :: ADNLPModel, x :: Vector, v :: Vector, Jtv :: Vector)
  nlp.counters.neval_jtprod += 1
  Jtv[1:nlp.meta.nvar] = ForwardDiff.jacobian(nlp.c, x)' * v
  return Jtv
end

function hess(nlp :: ADNLPModel, x :: Vector; obj_weight = 1.0, y :: Vector = [])
  nlp.counters.neval_hess += 1
  Hx = obj_weight == 0.0 ? spzeros(nlp.meta.nvar, nlp.meta.nvar) :
       ForwardDiff.hessian(nlp.f, x) * obj_weight
  for i = 1:length(y)
    if y[i] != 0.0
      Hx += ForwardDiff.hessian(x->nlp.c(x)[i], x) * y[i]
    end
  end
  return tril(Hx)
end

function hess_coord(nlp :: ADNLPModel, x :: Vector; obj_weight = 1.0, y :: Vector = [])
  Hx = sparse(hess(nlp, x, obj_weight=obj_weight, y=y))
  return findnz(Hx)
end

function hprod(nlp :: ADNLPModel, x :: Vector, v :: Vector;
    obj_weight = 1.0, y :: Vector = [])
  Hv = zeros(nlp.meta.nvar)
  return hprod!(nlp, x, v, Hv, obj_weight=obj_weight, y=y)
end

function hprod!(nlp :: ADNLPModel, x :: Vector, v :: Vector, Hv :: Vector;
    obj_weight = 1.0, y :: Vector = [])
  nlp.counters.neval_hprod += 1
  n = nlp.meta.nvar
  Hv[1:n] = obj_weight == 0.0 ? zeros(nlp.meta.nvar) :
          ForwardDiff.hessian(nlp.f, x) * v * obj_weight
  for i = 1:length(y)
    if y[i] != 0.0
      Hv[1:n] += ForwardDiff.hessian(x->nlp.c(x)[i], x) * v * y[i]
    end
  end
  return Hv
end
