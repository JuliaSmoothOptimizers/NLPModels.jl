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

function SimpleNLPModel(x0::Vector, obj::Function; y0::Vector = [], ncon = 0,
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
  if ncon == 0
    ncon = length(y0)
  else
    if length(y0) == 0
      y0 = zeros(ncon)
    else
      if length(y0) != ncon
        error("Passed ncon=$ncon and y0 with size $(length(y0))")
      end
    end
  end
  nvar = length(x0)
  length(lvar) == 0 && (lvar = -1e20*ones(nvar))
  length(uvar) == 0 && (uvar =  1e20*ones(nvar))
  if nnzh == 0
    try
      if ncon == 0
        A = hess(x0)
      else
        A = hess(x0,y0)
      end
      nnzh = typeof(A) <: SparseMatrixCSC ? nnz(A) : length(A)
    catch e
      if isa(e, NotImplementedError)
        nnzh = nvar^2
      else
        throw(e)
      end
    end
  end

  if ncon > 0
    length(lcon) == 0 && (lcon = -1e20*ones(ncon))
    length(ucon) == 0 && (ucon =  1e20*ones(ncon))
    if nnzj == 0
      try
        A = jac(x0)
        nnzj = typeof(A) <: SparseMatrixCSC ? nnz(A) : length(A)
      catch e
        if isa(e, NotImplementedError)
          nnzj = nvar*ncon
        else
          throw(e)
        end
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

function hess(nlp :: SimpleNLPModel, x :: Vector; y :: Vector = [])
  if length(y) == 0
    try
      nlp.H(x)
    catch
      nlp.H(x,zeros(nlp.meta.ncon))
    end
  else
    return nlp.H(x,y)
  end
end

function hess(nlp :: SimpleNLPModel, x :: Vector, y :: Vector)
  return nlp.H(x,y)
end

function hprod(nlp :: SimpleNLPModel, x :: Vector, v :: Vector; y :: Vector = [])
  try
    if length(y) == 0
      try
        return nlp.Hprod(x, v)
      catch
        return nlp.Hprod(x, zeros(nlp.meta.ncon), v)
      end
    else
      return nlp.Hprod(x, y, v)
    end
  catch e
    if isa(e, NotImplementedError)
      return hess(nlp, x, y=y)*v
    else
      throw(e)
    end
  end
end

hprod(nlp :: SimpleNLPModel, x :: Vector, y :: Vector, v :: Vector) =
  hprod(nlp, x, v, y=y)

function hprod!(nlp :: SimpleNLPModel, x :: Vector, v :: Vector, Hv :: Vector;
    y = [])
  try
    if length(y) == 0
      try
        return nlp.Hprod!(x, y, v, Hv)
      catch
        return nlp.Hprod!(x, zeros(nlp.meta.ncon), v, Hv)
      end
    else
      return nlp.Hprod!(x, y, v, Hv)
    end
  catch e
    if isa(e, NotImplementedError)
      w = hess(nlp, x, v, y=y)*v
      nw = length(w)
      for i = 1:nw
        Hv[i] = w[i]
      end
      return w
    else
      throw(e)
    end
  end
end

hprod!(nlp :: SimpleNLPModel, x :: Vector, y :: Vector, v :: Vector,
  Hv :: Vector) = hprod!(nlp, x, v, Hv, y=y)
