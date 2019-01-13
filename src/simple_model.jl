export SimpleNLPModel, obj, grad, grad!, cons, cons!, jac_coord, jac, jprod,
       jprod!, jtprod, jtprod!, hess, hprod, hprod!

"""SimpleNLPModel is an AbstractNLPModel that uses only user-defined functions.
In this interface, the objective function \$f\$ and an initial estimate are
required. If the user wants to use derivatives, they need to be passed. The
same goes for the Hessian and Hessian-AbstractVector product.
For constraints,
\$c:\\mathbb{R}^n\\rightarrow\\mathbb{R}^m\$  and the vectors
\$c_L\$ and \$c_U\$ also need to be passed. Bounds on the variables and an
inital estimate to the Lagrangian multipliers can also be provided.
The user can also pass the Jacobian and the Lagrangian Hessian and
Hessian-AbstractVector product.

````
SimpleNLPModel(f, x0; lvar = [-∞,…,-∞], uvar = [∞,…,∞], y0=zeros,
  lcon = [-∞,…,-∞], ucon = [∞,…,∞], name = "Generic",
  [list of functions])
````

  - `f :: Function` - The objective function \$f\$;
  - `x0 :: AbstractVector` - The initial point of the problem;
  - `lvar :: AbstractVector` - \$\\ell\$, the lower bound of the variables;
  - `uvar :: AbstractVector` - \$u\$, the upper bound of the variables;
  - `y0 :: AbstractVector` - The initial value of the Lagrangian estimates;
  - `lcon :: AbstractVector` - \$c_L\$, the lower bounds of the constraints function;
  - `ucon :: AbstractVector` - \$c_U\$, the upper bounds of the constraints function;
  - `name :: String` - A name for the model.

All functions passed have a direct correlation with a NLP function. You don't
have to define any more than you need, but calling an undefined function will
throw a `NotImplementedError`.
The list is

  - `g` and `g!`: \$\\nabla f(x)\$, the gradient of the objective function;

        gx = g(x)
        gx = g!(x, gx)

  - `H`: The lower triangle of the Hessian of the objective function or of the
    Lagrangian;

        Hx = H(x; obj_weight=1.0) # if the problem is unconstrained
        Hx = H(x; obj_weight=1.0, y=zeros) # if the problem is constrained

  - `Hcoord` - The lower triangle of the Hessian of the objective function
    or of the Lagrangian, in triplet format;

        (rows,cols,vals) = Hcoord(x; obj_weight=1.0) # if the problem is unconstrained
        (rows,cols,vals) = Hcoord(x; obj_weight=1.0, y=zeros) # if the problem is constrained

  - `Hp` and `Hp!` - The product of the Hessian of the objective function or of
    the Lagrangian by a vector;

        Hv = Hp(x, v, obj_weight=1.0) # if the problem is unconstrained
        Hv = Hp!(x, v, Hv, obj_weight=1.0) # if the problem is unconstrained
        Hv = Hp(x, v, obj_weight=1.0, y=zeros) # if the problem is constrained
        Hv = Hp!(x, v, Hv, obj_weight=1.0, y=zeros) # if the problem is constrained

  - `c` and `c!` - \$c(x)\$, the constraints function;

        cx = c(x)
        cx = c!(x, cx)

  - `J` - \$J(x)\$, the Jacobian of the constraints;

        Jx = J(x)

  - `Jcoord` - \$J(x)\$, the Jacobian of the constraints, in triplet format;

        (rows,cols,vals) = Jcoord(x)

  - `Jp` and `Jp!` - The Jacobian-vector product;

        Jv = Jp(x, v)
        Jv = Jp!(x, v, Jv)

  - `Jtp` and `Jtp!` - The Jacobian-transposed-vector product;

        Jtv = Jtp(x, v)
        Jtv = Jtp!(x, v, Jtv)

For contrained problems, the function \$c\$ is required, and it must return
an array even when m = 1,
and \$c_L\$ and \$c_U\$ should be passed, otherwise the problem is ill-formed.
For equality constraints, the corresponding index of \$c_L\$ and \$c_U\$ should be the
same.
"""
mutable struct SimpleNLPModel <: AbstractNLPModel
  meta :: NLPModelMeta

  counters :: Counters

  # Functions
  f :: Function
  g :: Function
  g! :: Function
  fg :: Function
  fg! :: Function
  H :: Function
  Hcoord :: Function
  Hp :: Function
  Hp! :: Function
  c :: Function
  c! :: Function
  fc :: Function
  fc! :: Function
  J :: Function
  Jcoord :: Function
  Jp :: Function
  Jp! :: Function
  Jtp :: Function
  Jtp! :: Function
end

NotImplemented(args...; kwargs...) = throw(NotImplementedError(""))

function SimpleNLPModel(f::Function, x0::AbstractVector; y0::AbstractVector = Float64[],
    lvar::AbstractVector = Float64[], uvar::AbstractVector = Float64[], lcon::AbstractVector = Float64[], ucon::AbstractVector = Float64[],
    nnzh::Int = 0, nnzj::Int = 0,
    g::Function = NotImplemented,
    g!::Function = NotImplemented,
    fg::Function = NotImplemented,
    fg!::Function = NotImplemented,
    H::Function = NotImplemented,
    Hcoord::Function = NotImplemented,
    Hp::Function = NotImplemented,
    Hp!::Function = NotImplemented,
    c::Function = NotImplemented,
    c!::Function = NotImplemented,
    fc::Function = NotImplemented,
    fc!::Function = NotImplemented,
    J::Function = NotImplemented,
    Jcoord::Function = NotImplemented,
    Jp::Function = NotImplemented,
    Jp!::Function = NotImplemented,
    Jtp::Function = NotImplemented,
    Jtp!::Function = NotImplemented,
    name::String = "Generic",
    lin::AbstractVector{Int} = Int[])

  nvar = length(x0)
  length(lvar) == 0 && (lvar = -Inf*ones(nvar))
  length(uvar) == 0 && (uvar =  Inf*ones(nvar))
  ncon = maximum([length(lcon); length(ucon); length(y0)])

  if ncon > 0
    length(lcon) == 0 && (lcon = -Inf*ones(ncon))
    length(ucon) == 0 && (ucon =  Inf*ones(ncon))
    length(y0) == 0   && (y0 = zeros(ncon))
  end

  nln = setdiff(1:ncon, lin)

  meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, y0=y0,
    lcon=lcon, ucon=ucon, nnzj=nnzj, nnzh=nnzh, name=name, lin=lin, nln=nln)

  return SimpleNLPModel(meta, Counters(), f, g, g!, fg, fg!, H, Hcoord, Hp,
                        Hp!, c, c!, fc, fc!, J, Jcoord, Jp, Jp!, Jtp, Jtp!)
end

function obj(nlp :: SimpleNLPModel, x :: AbstractVector)
  increment!(nlp, :neval_obj)
  return nlp.f(x)
end

function grad(nlp :: SimpleNLPModel, x :: AbstractVector)
  increment!(nlp, :neval_grad)
  return nlp.g(x)
end

function grad!(nlp :: SimpleNLPModel, x :: AbstractVector, g :: AbstractVector)
  increment!(nlp, :neval_grad)
  return nlp.g!(x, g)
end

function objgrad(nlp :: SimpleNLPModel, x :: AbstractVector)
  if nlp.fg == NotImplemented
    return obj(nlp, x), grad(nlp, x)
  else
    increment!(nlp, :neval_obj)
    increment!(nlp, :neval_grad)
    return nlp.fg(x)
  end
end

function objgrad!(nlp :: SimpleNLPModel, x :: AbstractVector, g :: AbstractVector)
  if nlp.fg! == NotImplemented
    return obj(nlp, x), grad!(nlp, x, g)
  else
    increment!(nlp, :neval_obj)
    increment!(nlp, :neval_grad)
    return nlp.fg!(x, g)
  end
end

function cons(nlp :: SimpleNLPModel, x :: AbstractVector)
  increment!(nlp, :neval_cons)
  return nlp.c(x)
end

function cons!(nlp :: SimpleNLPModel, x :: AbstractVector, c :: AbstractVector)
  increment!(nlp, :neval_cons)
  return nlp.c!(x, c)
end

function objcons(nlp :: SimpleNLPModel, x :: AbstractVector)
  if nlp.fc == NotImplemented
    return obj(nlp, x), nlp.meta.ncon > 0 ? cons(nlp, x) : Float64[]
  else
    increment!(nlp, :neval_obj)
    increment!(nlp, :neval_cons)
    return nlp.fc(x)
  end
end

function objcons!(nlp :: SimpleNLPModel, x :: AbstractVector, c :: AbstractVector)
  if nlp.fc! == NotImplemented
    return obj(nlp, x), nlp.meta.ncon > 0 ? cons!(nlp, x, c) : Float64[]
  else
    increment!(nlp, :neval_obj)
    increment!(nlp, :neval_cons)
    return nlp.fc!(x, c)
  end
end

function jac_coord(nlp :: SimpleNLPModel, x :: AbstractVector)
  increment!(nlp, :neval_jac)
  return nlp.Jcoord(x)
end

function jac(nlp :: SimpleNLPModel, x :: AbstractVector)
  increment!(nlp, :neval_jac)
  return nlp.J(x)
end

function jprod(nlp :: SimpleNLPModel, x :: AbstractVector, v :: AbstractVector)
  increment!(nlp, :neval_jprod)
  return nlp.Jp(x, v)
end

function jprod!(nlp :: SimpleNLPModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jprod)
  return nlp.Jp!(x, v, Jv)
end

function jtprod(nlp :: SimpleNLPModel, x :: AbstractVector, v :: AbstractVector)
  increment!(nlp, :neval_jtprod)
  return nlp.Jtp(x, v)
end

function jtprod!(nlp :: SimpleNLPModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod)
  return nlp.Jtp!(x, v, Jtv)
end

function hess(nlp :: SimpleNLPModel, x :: AbstractVector; obj_weight = 1.0,
      y :: AbstractVector = zeros(nlp.meta.ncon))
  increment!(nlp, :neval_hess)
  if nlp.meta.ncon > 0
    return nlp.H(x, obj_weight=obj_weight, y=y)
  else
    return nlp.H(x, obj_weight=obj_weight)
  end
end

function hess_coord(nlp :: SimpleNLPModel, x :: AbstractVector; obj_weight = 1.0,
      y :: AbstractVector = zeros(nlp.meta.ncon))
  increment!(nlp, :neval_hess)
  if nlp.meta.ncon > 0
    return nlp.Hcoord(x, obj_weight=obj_weight, y=y)
  else
    return nlp.Hcoord(x, obj_weight=obj_weight)
  end
end

function hprod(nlp :: SimpleNLPModel, x :: AbstractVector, v :: AbstractVector;
    obj_weight = 1.0, y :: AbstractVector = zeros(nlp.meta.ncon))
  increment!(nlp, :neval_hprod)
  if nlp.meta.ncon > 0
    return nlp.Hp(x, v, obj_weight=obj_weight, y=y)
  else
    return nlp.Hp(x, v; obj_weight=obj_weight)
  end
end

function hprod!(nlp :: SimpleNLPModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector;
    obj_weight = 1.0, y :: AbstractVector = zeros(nlp.meta.ncon))
  increment!(nlp, :neval_hprod)
  if nlp.meta.ncon > 0
    return nlp.Hp!(x, v, Hv, obj_weight=obj_weight, y=y)
  else
    return nlp.Hp!(x, v, Hv, obj_weight=obj_weight)
  end
end
