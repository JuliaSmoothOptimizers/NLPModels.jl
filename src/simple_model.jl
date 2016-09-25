import Compat.view

export SimpleNLPModel, obj, grad, grad!, cons, cons!, jac_coord, jac, jprod,
       jprod!, jtprod, jtprod!, hess, hprod, hprod!

"""SimpleNLPModel is an AbstractNLPModel that uses only user-defined functions.
In this interface, the objective function \$f\$ and an initial estimate are
required. If the user wants to use derivatives, they need to be passed. The
same goes for the Hessian and Hessian-Vector product.
For constraints,
\$c:\\mathbb{R}^n\\rightarrow\\mathbb{R}^m\$  and the vectors
\$c_L\$ and \$c_U\$ also need to be passed. Bounds on the variables and an
inital estimate to the Lagrangian multipliers can also be provided.
The user can also pass the Jacobian and the Lagrangian Hessian and
Hessian-Vector product.

````
SimpleNLPModel(f, x0; lvar = [-∞,…,-∞], uvar = [∞,…,∞], y0=zeros,
  lcon = [-∞,…,-∞], ucon = [∞,…,∞], name = "Generic",
  [list of functions])
````

  - `f :: Function` - The objective function \$f\$;
  - `x0 :: Vector` - The initial point of the problem;
  - `lvar :: Vector` - \$\\ell\$, the lower bound of the variables;
  - `uvar :: Vector` - \$u\$, the upper bound of the variables;
  - `y0 :: Vector` - The initial value of the Lagrangian estimates;
  - `lcon :: Vector` - \$c_L\$, the lower bounds of the constraints function;
  - `ucon :: Vector` - \$c_U\$, the upper bounds of the constraints function;
  - `name :: AbstractString` - A name for the model.

All functions passed have a direct correlation with a NLP function. You don't
have to define any more than you need, but calling an undefined function will
throw a `NotImplementedError`.
The list is

  - `g` and `g!`: \$\\nabla f(x)\$, the gradient of the objective function;
    see [grad](api/#grad).

    gx = g(x)
    gx = g!(x, gx)

  - `H`: The lower triangle of the Hessian of the objective function or of the
    Lagrangian;
    see [hess](api/#hess).

    Hx = H(x; obj_weight=1.0) # if the problem is unconstrained
    Hx = H(x; obj_weight=1.0, y=zeros) # if the problem is constrained

  - `Hcoord` - The lower triangle of the Hessian of the objective function
    or of the Lagrangian, in triplet format;
    see [hess_coord](api/#hess_coord).

    (rows,cols,vals) = Hcoord(x; obj_weight=1.0) # if the problem is unconstrained
    (rows,cols,vals) = Hcoord(x; obj_weight=1.0, y=zeros) # if the problem is constrained

  - `Hp` and `Hp!` - The product of the Hessian of the objective function or of
    the Lagrangian by a vector;
    see [hprod](api/#hprod).

    Hv = Hp(x, v, obj_weight=1.0) # if the problem is unconstrained
    Hv = Hp!(x, v, Hv, obj_weight=1.0) # if the problem is unconstrained
    Hv = Hp(x, v, obj_weight=1.0, y=zeros) # if the problem is constrained
    Hv = Hp!(x, v, Hv, obj_weight=1.0, y=zeros) # if the problem is constrained

  - `c` and `c!` - \$c(x)\$, the constraints function;
    see [cons](api/#cons).

    cx = c(x)
    cx = c!(x, cx)

  - `J` - \$J(x)\$, the Jacobian of the constraints;
    see [jac](api/#jac).

    Jx = J(x)

  - `Jcoord` - \$J(x)\$, the Jacobian of the constraints, in triplet format;
    see [jac_coord](api/#jac_coord).

    (rows,cols,vals) = Jcoord(x)

  - `Jp` and `Jp!` - The Jacobian-vector product;
    see [jprod](api/#jprod).

    Jv = Jp(x, v)
    Jv = Jp!(x, v, Jv)

  - `Jtp` and `Jtp!` - The Jacobian-transposed-vector product;
    see [jtprod](api/#jtprod).

    Jtv = Jtp(x, v)
    Jtv = Jtp!(x, v, Jtv)

For contrained problems, the function \$c\$ is required, and it must return
an array even when m = 1,
and \$c_L\$ and \$c_U\$ should be passed, otherwise the problem is ill-formed.
For equality constraints, the corresponding index of \$c_L\$ and \$c_U\$ should be the
same.
"""
type SimpleNLPModel <: AbstractNLPModel
  meta :: NLPModelMeta

  counters :: Counters

  # Functions
  f :: Function
  g :: Function
  g! :: Function
  H :: Function
  Hcoord :: Function
  Hp :: Function
  Hp! :: Function
  c :: Function
  c! :: Function
  J :: Function
  Jcoord :: Function
  Jp :: Function
  Jp! :: Function
  Jtp :: Function
  Jtp! :: Function
end

NotImplemented(args...; kwargs...) = throw(NotImplementedError(""))

function SimpleNLPModel(f::Function, x0::Vector; y0::Vector = [],
    lvar::Vector = [], uvar::Vector = [], lcon::Vector = [], ucon::Vector = [],
    nnzh::Int = 0, nnzj::Int = 0,
    g::Function = NotImplemented,
    g!::Function = NotImplemented,
    H::Function = NotImplemented,
    Hcoord::Function = NotImplemented,
    Hp::Function = NotImplemented,
    Hp!::Function = NotImplemented,
    c::Function = NotImplemented,
    c!::Function = NotImplemented,
    J::Function = NotImplemented,
    Jcoord::Function = NotImplemented,
    Jp::Function = NotImplemented,
    Jp!::Function = NotImplemented,
    Jtp::Function = NotImplemented,
    Jtp!::Function = NotImplemented,
    name::AbstractString = "Generic")

  nvar = length(x0)
  length(lvar) == 0 && (lvar = -Inf*ones(nvar))
  length(uvar) == 0 && (uvar =  Inf*ones(nvar))
  ncon = maximum([length(lcon); length(ucon); length(y0)])

  if ncon > 0
    length(lcon) == 0 && (lcon = -Inf*ones(ncon))
    length(ucon) == 0 && (ucon =  Inf*ones(ncon))
    length(y0) == 0   && (y0 = zeros(ncon))
  end

  meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, y0=y0,
    lcon=lcon, ucon=ucon, nnzj=nnzj, nnzh=nnzh, name=name)

  return SimpleNLPModel(meta, Counters(), f, g, g!, H, Hcoord, Hp, Hp!, c, c!,
      J, Jcoord, Jp, Jp!, Jtp, Jtp!)
end

function obj(nlp :: SimpleNLPModel, x :: Vector)
  nlp.counters.neval_obj += 1
  return nlp.f(x)
end

function grad(nlp :: SimpleNLPModel, x :: Vector)
  nlp.counters.neval_grad += 1
  return nlp.g(x)
end

function grad!(nlp :: SimpleNLPModel, x :: Vector, g :: Vector)
  nlp.counters.neval_grad += 1
  return nlp.g!(x, g)
end

function cons(nlp :: SimpleNLPModel, x :: Vector)
  nlp.counters.neval_cons += 1
  return nlp.c(x)
end

function cons!(nlp :: SimpleNLPModel, x :: Vector, c :: Vector)
  nlp.counters.neval_cons += 1
  return nlp.c!(x, c)
end

function jac_coord(nlp :: SimpleNLPModel, x :: Vector)
  nlp.counters.neval_jac += 1
  return nlp.Jcoord(x)
end

function jac(nlp :: SimpleNLPModel, x :: Vector)
  nlp.counters.neval_jac += 1
  return nlp.J(x)
end

function jprod(nlp :: SimpleNLPModel, x :: Vector, v :: Vector)
  nlp.counters.neval_jprod += 1
  return nlp.Jp(x, v)
end

function jprod!(nlp :: SimpleNLPModel, x :: Vector, v :: Vector, Jv :: Vector)
  nlp.counters.neval_jprod += 1
  return nlp.Jp!(x, v, Jv)
end

function jtprod(nlp :: SimpleNLPModel, x :: Vector, v :: Vector)
  nlp.counters.neval_jtprod += 1
  return nlp.Jtp(x, v)
end

function jtprod!(nlp :: SimpleNLPModel, x :: Vector, v :: Vector, Jtv :: Vector)
  nlp.counters.neval_jtprod += 1
  return nlp.Jtp!(x, v, Jtv)
end

function hess(nlp :: SimpleNLPModel, x :: Vector; obj_weight = 1.0,
      y :: Vector = zeros(nlp.meta.ncon))
  nlp.counters.neval_hess += 1
  if nlp.meta.ncon > 0
    return nlp.H(x, obj_weight=obj_weight, y=y)
  else
    return nlp.H(x, obj_weight=obj_weight)
  end
end

function hess_coord(nlp :: SimpleNLPModel, x :: Vector; obj_weight = 1.0,
      y :: Vector = zeros(nlp.meta.ncon))
  nlp.counters.neval_hess += 1
  if nlp.meta.ncon > 0
    return nlp.Hcoord(x, obj_weight=obj_weight, y=y)
  else
    return nlp.Hcoord(x, obj_weight=obj_weight)
  end
end

function hprod(nlp :: SimpleNLPModel, x :: Vector, v :: Vector;
    obj_weight = 1.0, y :: Vector = zeros(nlp.meta.ncon))
  nlp.counters.neval_hprod += 1
  if nlp.meta.ncon > 0
    return nlp.Hp(x, v, obj_weight=obj_weight, y=y)
  else
    return nlp.Hp(x, v; obj_weight=obj_weight)
  end
end

function hprod!(nlp :: SimpleNLPModel, x :: Vector, v :: Vector, Hv :: Vector;
    obj_weight = 1.0, y :: Vector = zeros(nlp.meta.ncon))
  nlp.counters.neval_hprod += 1
  if nlp.meta.ncon > 0
    return nlp.Hp!(x, v, Hv, obj_weight=obj_weight, y=y)
  else
    return nlp.Hp!(x, v, Hv, obj_weight=obj_weight)
  end
end
