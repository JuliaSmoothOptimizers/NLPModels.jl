export SimpleNLSModel,
       residual, residual!, jac_residual, jprod_residual, jprod_residual!,
       jtprod_residual, jtprod_residual!, jac_op_residual, hess_residual,
       hprod_residual, hprod_residual!

"""
    nls = SimpleNLSModel(n;  F=F, F! =F!, JF=JF, JFp=JFp, JFp! =JFp!,
    JFtp=JFtp, JFtp! =JFtp!)
    nls = SimpleNLSModel(x0; F=F, F! =F!, JF=JF, JFp=JFp, JFp! =JFp!,
    JFtp=JFtp, JFtp! =JFtp!)

Creates a Nonlinear Linear Least Squares model to minimize ‖F(x)‖².
If JF = JF(x) is passed, the Jacobian is available.
"""
mutable struct SimpleNLSModel <: AbstractNLSModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  counters :: NLSCounters

  F :: Function
  F! :: Function
  JF :: Function
  JFp :: Function
  JFp! :: Function
  JFtp :: Function
  JFtp! :: Function
  Hi :: Function # Hi(x, i) return the Hessian (tri-inf)
  Hip :: Function # Hip(x, i, v) returns the product
  Hip! :: Function # Hip!(x, i, v, Hiv) computes the product
  c :: Function
  c! :: Function
  J :: Function
  Jcoord :: Function
  Jp :: Function
  Jp! :: Function
  Jtp :: Function
  Jtp! :: Function
  Hc :: Function # Hessian for the constraints
  Hcp :: Function
  Hcp! :: Function
end

function SimpleNLSModel(x0 :: AbstractVector, nequ :: Int;
                        lvar :: AbstractVector = fill(-Inf, length(x0)),
                        uvar :: AbstractVector = fill(Inf, length(x0)),
                        lcon :: AbstractVector = Float64[],
                        ucon :: AbstractVector = Float64[],
                        y0 :: AbstractVector = zeros(max(length(lcon), length(ucon))),
                        F :: Function = NotImplemented,
                        F! :: Function = NotImplemented,
                        JF :: Function = NotImplemented,
                        JFp :: Function = NotImplemented,
                        JFp! :: Function = NotImplemented,
                        JFtp :: Function = NotImplemented,
                        JFtp! :: Function = NotImplemented,
                        Hi :: Function = NotImplemented,
                        Hip :: Function = NotImplemented,
                        Hip! :: Function = NotImplemented,
                        c::Function = NotImplemented,
                        c!::Function = NotImplemented,
                        J::Function = NotImplemented,
                        Jcoord::Function = NotImplemented,
                        Jp::Function = NotImplemented,
                        Jp!::Function = NotImplemented,
                        Jtp::Function = NotImplemented,
                        Jtp!::Function = NotImplemented,
                        Hc :: Function = NotImplemented,
                        Hcp :: Function = NotImplemented,
                        Hcp! :: Function = NotImplemented
                       )
  nvar = length(x0)
  ncon = maximum([length(lcon); length(ucon); length(y0)])
  if !(length(lcon) == length(ucon) == length(y0))
    error("lcon, ucon and y0 need to be the same length")
  end
  nnzj = nvar * ncon

  meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon,
                      y0=y0, lcon=lcon, ucon=ucon, nnzj=nnzj)
  nls_meta = NLSMeta(nequ, nvar)
  return SimpleNLSModel(meta, nls_meta, NLSCounters(), F, F!, JF, JFp,
                        JFp!, JFtp, JFtp!, Hi, Hip, Hip!, c, c!, J,
                        Jcoord, Jp, Jp!, Jtp, Jtp!, Hc, Hcp, Hcp!)
end

SimpleNLSModel(nvar :: Int, nequ :: Int; kwargs...) = SimpleNLSModel(zeros(nvar), nequ; kwargs...)

function residual(nls :: SimpleNLSModel, x :: AbstractVector)
  increment!(nls, :neval_residual)
  return nls.F(x)
end

function residual!(nls :: SimpleNLSModel, x :: AbstractVector, Fx :: AbstractVector)
  increment!(nls, :neval_residual)
  nls.F!(x, Fx)
  return Fx
end

function jac_residual(nls :: SimpleNLSModel, x :: AbstractVector)
  increment!(nls, :neval_jac_residual)
  return nls.JF(x)
end

function jprod_residual(nls :: SimpleNLSModel, x :: AbstractVector, v :: AbstractVector)
  increment!(nls, :neval_jprod_residual)
  return nls.JFp(x, v)
end

function jprod_residual!(nls :: SimpleNLSModel, x :: AbstractVector, v
                         :: AbstractVector, JFv :: AbstractVector)
  increment!(nls, :neval_jprod_residual)
  nls.JFp!(x, v, JFv)
  return JFv
end

function jtprod_residual(nls :: SimpleNLSModel, x :: AbstractVector, v :: AbstractVector)
  increment!(nls, :neval_jtprod_residual)
  return nls.JFtp(x, v)
end

function jtprod_residual!(nls :: SimpleNLSModel, x :: AbstractVector, v
                          :: AbstractVector, JFtv :: AbstractVector)
  increment!(nls, :neval_jtprod_residual)
  nls.JFtp!(x, v, JFtv)
  return JFtv
end

function hess_residual(nls :: SimpleNLSModel, x :: AbstractVector, i :: Int)
  increment!(nls, :neval_hess_residual)
  return nls.Hi(x, i)
end

function hprod_residual(nls :: SimpleNLSModel, x :: AbstractVector, i :: Int, v :: AbstractVector)
  increment!(nls, :neval_hprod_residual)
  return nls.Hip(x, i, v)
end

function hprod_residual!(nls :: SimpleNLSModel, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  increment!(nls, :neval_hprod_residual)
  nls.Hip!(x, i, v, Hiv)
  return Hiv
end

function cons(nls :: SimpleNLSModel, x :: AbstractVector)
  increment!(nls, :neval_cons)
  return nls.c(x)
end

function cons!(nls :: SimpleNLSModel, x :: AbstractVector, c :: AbstractVector)
  increment!(nls, :neval_cons)
  return nls.c!(x, c)
end

function jac_coord(nls :: SimpleNLSModel, x :: AbstractVector)
  increment!(nls, :neval_jac)
  return nls.Jcoord(x)
end

function jac(nls :: SimpleNLSModel, x :: AbstractVector)
  increment!(nls, :neval_jac)
  return nls.J(x)
end

function jprod(nls :: SimpleNLSModel, x :: AbstractVector, v :: AbstractVector)
  increment!(nls, :neval_jprod)
  return nls.Jp(x, v)
end

function jprod!(nls :: SimpleNLSModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nls, :neval_jprod)
  return nls.Jp!(x, v, Jv)
end

function jtprod(nls :: SimpleNLSModel, x :: AbstractVector, v :: AbstractVector)
  increment!(nls, :neval_jtprod)
  return nls.Jtp(x, v)
end

function jtprod!(nls :: SimpleNLSModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nls, :neval_jtprod)
  return nls.Jtp!(x, v, Jtv)
end

function hess(nls :: SimpleNLSModel, x :: AbstractVector; obj_weight = 1.0, y :: AbstractVector = Float64[])
  increment!(nls, :neval_hess)
  Fx = residual(nls, x)
  Jx = jac_residual(nls, x)
  Hx = obj_weight == 0.0 ? spzeros(nls.meta.nvar, nls.meta.nvar) : Jx' * Jx * obj_weight
  if obj_weight != 0.0
    m = length(Fx)
    for i = 1:m
      Hx += obj_weight * Fx[i] * hess_residual(nls, x, i)
    end
  end
  if length(y) > 0
    Hx += nls.Hc(x, y)
  end
  return tril(Hx)
end

function hess_coord(nls :: SimpleNLSModel, x :: AbstractVector; obj_weight = 1.0, y :: AbstractVector = Float64[])
  H = hess(nls, x, obj_weight=obj_weight, y=y)
  return findnz(H)
end

function hprod(nls :: SimpleNLSModel, x :: AbstractVector, v :: AbstractVector;
    obj_weight = 1.0, y :: AbstractVector = Float64[])
  increment!(nls, :neval_hprod)
  n = nls.meta.nvar
  Hv = length(y) > 0 ? nls.Hcp(x, y, v) : zeros(n)
  if obj_weight != 0.0
    Fx = residual(nls, x)
    Jv = jprod_residual(nls, x, v)
    @views Hv[1:n] .+= obj_weight * jtprod_residual(nls, x, Jv)
    m = length(Fx)
    Hiv = zeros(n)
    for i = 1:m
      hprod_residual!(nls, x, i, v, Hiv)
      @views Hv[1:n] .+= (obj_weight * Fx[i]) * Hiv
    end
  end
  return Hv
end

function hprod!(nls :: SimpleNLSModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector;
    obj_weight = 1.0, y :: AbstractVector = Float64[])
  increment!(nls, :neval_hprod)
  n = nls.meta.nvar
  if length(y) > 0
    nls.Hcp!(x, y, v, Hv)
  else
    @views fill!(Hv[1:n], 0.0)
  end
  if obj_weight != 0.0
    Fx = residual(nls, x)
    Jv = jprod_residual(nls, x, v)
    @views Hv[1:n] .+= obj_weight * jtprod_residual(nls, x, Jv)
    m = length(Fx)
    Hiv = zeros(n)
    for i = 1:m
      hprod_residual!(nls, x, i, v, Hiv)
      @views Hv[1:n] .+= (obj_weight * Fx[i]) * Hiv
    end
  end
  return Hv
end
