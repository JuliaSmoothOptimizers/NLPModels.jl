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
  length(lvar) == 0 && (lvar = -1e20*ones(nvar))
  length(uvar) == 0 && (uvar =  1e20*ones(nvar))
  if nnzh == 0
    try
      A = hess(x0)
      nnzh = typeof(A) <: SparseMatrixCSC ? nnz(A) : length(A)
    catch e
      if isa(e, NotImplementedError)
        nnzh = nvar^2
      else
        throw(e)
      end
    end
  end

  ncon = length(y0)
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
  return nlp.g(x)
end

function grad!(nlp :: SimpleNLPModel, x :: Vector, g :: Vector)
  nlp.g!(x, g)
  return g
end

function cons(nlp :: SimpleNLPModel, x :: Vector)
  return nlp.c(x)
end

function cons!(nlp :: SimpleNLPModel, x :: Vector, c :: Vector)
  nlp.c!(x, c)
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

function hess(nlp :: SimpleNLPModel, x :: Vector)
  return nlp.H(x)
end

function hprod(nlp :: SimpleNLPModel, x :: Vector, v :: Vector)
  try
    return nlp.Hprod(x, v)
  catch e
    if isa(e, NotImplementedError)
      return nlp.H(x)*v
    else
      throw(e)
    end
  end
end

function hprod!(nlp :: SimpleNLPModel, x :: Vector, v :: Vector, Hv :: Vector)
  try
    return nlp.Hprod(x, v)
  catch e
    if isa(e, NotImplementedError)
      w = nlp.H(x)*v
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

