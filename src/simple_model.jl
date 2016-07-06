using ForwardDiff

export SimpleNLPModel, obj, grad, grad!, cons, cons!, jac_coord, jac, jprod,
       jprod!, jtprod, jtprod!, hess, hprod, hprod!

type SimpleNLPModel <: AbstractNLPModel
  meta :: NLPModelMeta

  counters :: Counters

  # Functions
  f :: Function
  c :: Function
end

function SimpleNLPModel(x0::Vector, f::Function; y0::Vector = [],
    lvar::Vector = [], uvar::Vector = [], lcon::Vector = [], ucon::Vector = [],
    c::Function = (args...)->throw(NotImplementedError("cons")))

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
    islp=false)

  return SimpleNLPModel(meta, Counters(), f, c)
end

function obj(nlp :: SimpleNLPModel, x :: Vector)
  return nlp.f(x)
end

function grad(nlp :: SimpleNLPModel, x :: Vector)
  return ForwardDiff.gradient(nlp.f, x)
end

function grad!(nlp :: SimpleNLPModel, x :: Vector, g :: Vector)
  return ForwardDiff.gradient!(g, nlp.f, x)
end

function cons(nlp :: SimpleNLPModel, x :: Vector)
  return nlp.c(x)
end

function cons!(nlp :: SimpleNLPModel, x :: Vector, c :: Vector)
  c[:] = nlp.c(x)
  return c
end

function jac_coord(nlp :: SimpleNLPModel, x :: Vector)
  J = ForwardDiff.jacobian(nlp.c, x)
  return typeof(J) <: Matrix ? findnz(sparse(J)) : findnz(J)
end

function jac(nlp :: SimpleNLPModel, x :: Vector)
  return ForwardDiff.jacobian(nlp.c, x)
end

function jprod(nlp :: SimpleNLPModel, x :: Vector, v :: Vector)
  return ForwardDiff.jacobian(nlp.c, x) * v
end

function jprod!(nlp :: SimpleNLPModel, x :: Vector, v :: Vector, Jv :: Vector)
  Jv[:] = ForwardDiff.jacobian(nlp.c, x) * v
  return Jv
end

function jtprod(nlp :: SimpleNLPModel, x :: Vector, v :: Vector)
  return ForwardDiff.jacobian(nlp.c, x)' * v
end

function jtprod!(nlp :: SimpleNLPModel, x :: Vector, v :: Vector, Jtv :: Vector)
  Jtv[:] = ForwardDiff.jacobian(nlp.c, x)' * v
  return Jtv
end

function hess(nlp :: SimpleNLPModel, x :: Vector; obj_weight = 1.0, y :: Vector = [])
  Hx = ForwardDiff.hessian(nlp.f, x) * obj_weight
  for i = 1:length(y)
    Hx += ForwardDiff.hessian(x->nlp.c(x)[i], x) * y[i]
  end
  return tril(Hx)
end

function hprod(nlp :: SimpleNLPModel, x :: Vector, v :: Vector;
    obj_weight = 1.0, y :: Vector = [])
  Hv = ForwardDiff.hessian(nlp.f, x) * v * obj_weight
  for i = 1:length(y)
    Hv += ForwardDiff.hessian(x->nlp.c(x)[i], x) * v * y[i]
  end
  return Hv
end

function hprod!(nlp :: SimpleNLPModel, x :: Vector, v :: Vector, Hv :: Vector;
    obj_weight = 1.0, y :: Vector = [])
  Hv[:] = ForwardDiff.hessian(nlp.f, x) * v * obj_weight
  for i = 1:length(y)
    Hv[:] += ForwardDiff.hessian(x->nlp.c(x)[i], x) * v * y[i]
  end
  return Hv
end
