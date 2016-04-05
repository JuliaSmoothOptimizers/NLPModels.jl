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
  c :: Function
  c! :: Function
  J :: Function
end

function SimpleNLPModel(x0; y0 = [], lvar = [], uvar = [], lcon = [], ucon = [],
    nnzj = 0, nnzh = 0, obj = error, grad = error, grad! = error, hess =
    error, cons = error, cons! = error, jac = error)
  nvar = length(x0)
  length(lvar) == 0 && (lvar = -1e20*ones(nvar))
  length(uvar) == 0 && (uvar =  1e20*ones(nvar))
  if nnzh == 0
    if hess != error
      A = hess(x0)
      nnzh = typeof(A) <: SparseMatrixCSC ? nnz(A) : length(A)
    else
      nnzh = nvar^2
    end
  end

  ncon = length(y0)
  if ncon > 0
    length(lcon) == 0 && (lcon = -1e20*ones(ncon))
    length(ucon) == 0 && (ucon =  1e20*ones(ncon))
    if nnzj == 0
      if jac != error
        A = jac(x0)
        nnzj = typeof(A) <: SparseMatrixCSC ? nnz(A) : length(A)
      else
        nnzj = nvar*ncon
      end
    end
  end
  lin = []
  nln = collect(1:ncon)

  meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, y0=y0,
    lcon=lcon, ucon=ucon, nnzj=nnzj, nnzh=nnzh, lin=lin, nln=nln, minimize=true,
    islp=false)

  return SimpleNLPModel(meta, Counters(), obj, grad, grad!, hess, cons, cons!, jac)
end

function obj(nlp :: SimpleNLPModel, x)
  return nlp.f(x)
end

function grad(nlp :: SimpleNLPModel, x)
  return nlp.g(x)
end

function grad!(nlp :: SimpleNLPModel, x, g)
  nlp.g!(x, g)
  return g
end

function cons(nlp :: SimpleNLPModel, x)
  return nlp.c(x)
end

function cons!(nlp :: SimpleNLPModel, x, c)
  nlp.c!(x, c)
  return c
end

function jac_coord(nlp :: SimpleNLPModel, x)
  J = nlp.J(x)
  if J <: Matrix
    rows = Int[]
    cols = Int[]
    vals = []
    for j = 1:nlp.meta.nvar
      for i = 1:nlp.meta.ncon
        a = J[i,j]
        if a != 0.0
          push!(vals, a)
          push!(rows, i)
          push!(cols, j)
        end
      end
    end
    return rows, cols, vals
  else
    rows = rowvals(J)
    vals = nonzeros(A)
    cols = Int[]
    for i = 1:nlp.meta.nvar
      append!(nzrange(J, i))
    end
    return rows, cols, vals
  end
end

function jac(nlp :: SimpleNLPModel, x)
  return nlp.J(x)
end

function jprod(nlp :: SimpleNLPModel, x, v)
  return nlp.J(x)*v
end

function jprod!(nlp :: SimpleNLPModel, x, v, Jv)
  w = nlp.J(x)*v
  nw = length(w)
  for i = 1:nw
    Jv[i] = w[i]
  end
  return w
end

function jtprod(nlp :: SimpleNLPModel, x, v)
  return nlp.J(x)'*v
end

function jtprod!(nlp :: SimpleNLPModel, x, v, Jtv)
  w = nlp.J(x)'*v
  nw = length(w)
  for i = 1:nw
    Jtv[i] = w[i]
  end
  return w
end

function hess(nlp :: SimpleNLPModel, x)
  return nlp.H(x)
end

function hprod(nlp :: SimpleNLPModel, x, v)
  return nlp.H(x)*v
end

function hprod!(nlp :: SimpleNLPModel, x, v, Hv)
  w = nlp.H(x)'*v
  nw = length(w)
  for i = 1:nw
    Hv[i] = w[i]
  end
  return w
end

