export SimpleNLPModel, obj, grad, grad!, cons, cons!, jac_coord, jac, jprod,
       jprod!, jtprod, jtprod!, hess, hprod, hprod!

type SimpleNLPModel <: AbstractNLPModel
  meta :: NLPModelMeta

  counters :: Counters

  # Functions
  f :: Function
  g :: Function
  g! :: Function
  H :: Function
  Hprod :: Function
  Hprod! :: Function
  c :: Function
  c! :: Function
  J :: Function
  Jprod :: Function
  Jprod! :: Function
  Jtprod :: Function
  Jtprod! :: Function
end

function SimpleNLPModel(x0::Vector, obj::Function; y0::Vector = [],
    lvar::Vector = [], uvar::Vector = [], lcon::Vector = [], ucon::Vector = [],
    nnzj::Int = 0, nnzh::Int = 0,
    grad::Function = (args...)->throw(NotImplementedError("grad")),
    grad!::Function = (args...)->throw(NotImplementedError("grad!")),
    hess::Function = (args...)->throw(NotImplementedError("hess")),
    hprod::Function = (args...)->throw(NotImplementedError("hprod")),
    hprod!::Function = (args...)->throw(NotImplementedError("hprod!")),
    cons::Function = (args...)->throw(NotImplementedError("cons")),
    cons!::Function = (args...)->throw(NotImplementedError("cons!")),
    jac::Function = (args...)->throw(NotImplementedError("jac")),
    jprod::Function = (args...)->throw(NotImplementedError("jprod")),
    jprod!::Function = (args...)->throw(NotImplementedError("jprod!")),
    jtprod::Function = (args...)->throw(NotImplementedError("jtprod")),
    jtprod!::Function = (args...)->throw(NotImplementedError("jtprod!")))

  nvar = length(x0)
  length(lvar) == 0 && (lvar = -Inf*ones(nvar))
  length(uvar) == 0 && (uvar =  Inf*ones(nvar))
  ncon = maximum([length(lcon); length(ucon); length(y0)])
  if nnzh == 0
    try
      if ncon == 0
        A = hess(x0)
      else
        A = hess(x0, y0)
      end
      nnzh = typeof(A) <: SparseMatrixCSC ? nnz(A) : length(A)
    end
  end

  if ncon > 0
    length(lcon) == 0 && (lcon = -Inf*ones(ncon))
    length(ucon) == 0 && (ucon =  Inf*ones(ncon))
    length(y0) == 0   && (y0 = zeros(ncon))
    if nnzj == 0
      try
        A = jac(x0)
        nnzj = typeof(A) <: SparseMatrixCSC ? nnz(A) : length(A)
      end
    end
  end
  lin = []
  nln = collect(1:ncon)

  meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, y0=y0,
    lcon=lcon, ucon=ucon, nnzj=nnzj, nnzh=nnzh, lin=lin, nln=nln, minimize=true,
    islp=false)

  return SimpleNLPModel(meta, Counters(), obj, grad, grad!, hess, hprod, hprod!,
    cons, cons!, jac, jprod, jprod!, jtprod, jtprod!)
end

function obj(nlp :: SimpleNLPModel, x :: Vector)
  return nlp.f(x)
end

function grad(nlp :: SimpleNLPModel, x :: Vector)
  try
    return nlp.g(x)
  catch e
    if isa(e, NotImplementedError)
      g = zeros(x)
      nlp.g!(x, g)
      return g
    else
      throw(e)
    end
  end
end

function grad!(nlp :: SimpleNLPModel, x :: Vector, g :: Vector)
  try
    nlp.g!(x, g)
  catch e
    if isa(e, NotImplementedError)
      gp = nlp.g(x)
      for i = 1:nlp.meta.nvar
        g[i] = gp[i]
      end
    else
      throw(e)
    end
  end
  return g
end

function cons(nlp :: SimpleNLPModel, x :: Vector)
  try
    return nlp.c(x)
  catch e
    if isa(e, NotImplementedError)
      c = zeros(nlp.meta.ncon)
      nlp.c!(x, c)
      return c
    else
      throw(e)
    end
  end
end

function cons!(nlp :: SimpleNLPModel, x :: Vector, c :: Vector)
  try
    nlp.c!(x, c)
  catch e
    if isa(e, NotImplementedError)
      cp = nlp.c(x)
      for i = 1:nlp.meta.ncon
        c[i] = cp[i]
      end
    else
      throw(e)
    end
  end
  return c
end

function jac_coord(nlp :: SimpleNLPModel, x :: Vector)
  J = nlp.J(x)
  return typeof(J) <: Matrix ? findnz(sparse(J)) : findnz(J)
end

function jac(nlp :: SimpleNLPModel, x :: Vector)
  return nlp.J(x)
end

function jprod(nlp :: SimpleNLPModel, x :: Vector, v :: Vector)
  try
    return nlp.Jprod(x, v)
  catch e
    if isa(e, NotImplementedError)
      return nlp.J(x)*v
    else
      throw(e)
    end
  end
end

function jprod!(nlp :: SimpleNLPModel, x :: Vector, v :: Vector, Jv :: Vector)
  try
    nlp.Jprod!(x, v, Jv)
    return Jv
  catch e
    if isa(e, NotImplementedError)
      w = nlp.J(x)*v
      nw = length(w)
      for i = 1:nw
        Jv[i] = w[i]
      end
      return w
    else
      throw(e)
    end
  end
end

function jtprod(nlp :: SimpleNLPModel, x :: Vector, v :: Vector)
  try
    return nlp.Jtprod(x, v)
  catch e
    if isa(e, NotImplementedError)
      return nlp.J(x)'*v
    else
      throw(e)
    end
  end
end

function jtprod!(nlp :: SimpleNLPModel, x :: Vector, v :: Vector, Jtv :: Vector)
  try
    nlp.Jtprod!(x, v, Jtv)
    return Jtj
  catch e
    if isa(e, NotImplementedError)
      w = nlp.J(x)'*v
      nw = length(w)
      for i = 1:nw
        Jtv[i] = w[i]
      end
      return w
    else
      throw(e)
    end
  end
end

function hess(nlp :: SimpleNLPModel, x :: Vector; obj_weight = 1.0, y :: Vector = [])
  if length(y) == 0
    try
      return tril(nlp.H(x)) * obj_weight
    catch
      return tril(nlp.H(x,zeros(nlp.meta.ncon))) * obj_weight
    end
  else
    if obj_weight != 0.0
      return tril(nlp.H(x,y/obj_weight)) * obj_weight
    else
      return tril(nlp.H(x,y) - nlp.H(x,zeros(y)))
    end
  end
end

function hess(nlp :: SimpleNLPModel, x :: Vector, y :: Vector)
  return tril(nlp.H(x,y))
end

function hprod(nlp :: SimpleNLPModel, x :: Vector, v :: Vector; y :: Vector = [])
  if nlp.meta.ncon == 0
    try
      return nlp.Hprod(x, v)
    catch
      return nlp.H(x) * v
    end
  else
    w = length(y) == 0 ? zeros(nlp.meta.ncon) : y
    try
      return nlp.Hprod(x, w, v)
    catch
      return nlp.H(x,w) * v
    end
  end
end

hprod(nlp :: SimpleNLPModel, x :: Vector, y :: Vector, v :: Vector) =
  hprod(nlp, x, v, y=y)

function hprod!(nlp :: SimpleNLPModel, x :: Vector, v :: Vector, Hv :: Vector;
    y = [])
  if nlp.meta.ncon == 0
    try
      return nlp.Hprod!(x, v, Hv)
    catch
      Hv[:] = nlp.H(x) * v
      return Hv
    end
  else
    w = length(y) == 0 ? zeros(nlp.meta.ncon) : y
    try
      return nlp.Hprod!(x, w, v, Hv)
    catch
      Hv[:] = nlp.H(x,w) * v
      return Hv
    end
  end
end

hprod!(nlp :: SimpleNLPModel, x :: Vector, y :: Vector, v :: Vector,
  Hv :: Vector) = hprod!(nlp, x, v, Hv, y=y)
