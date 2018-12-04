using NLPModels: increment!

#Problem 6 in the Hock-Schittkowski suite
function hs6_autodiff()
  x0 = [-1.2; 1.0]
  f(x) = (1 - x[1])^2
  c(x) = [10 * (x[2] - x[1]^2)]
  lcon = [0.0]
  ucon = [0.0]

  return ADNLPModel(f, x0, c=c, lcon=lcon, ucon=ucon)
end
hs6so_autodiff = hs6_autodiff
function hs6nls_autodiff()
  x0 = [-1.2; 1.0]
  F(x) = [1 - x[1]]
  c(x) = [10 * (x[2] - x[1]^2)]
  lcon = [0.0]
  ucon = [0.0]

  return ADNLPModel(F, 1, x0, σnls=2.0, c=c, lcon=lcon, ucon=ucon)
end
function hs6ls_autodiff()
  x0 = [-1.2; 1.0]
  A, b = [-1.0 0.0], [-1.0]
  c(x) = [10 * (x[2] - x[1]^2)]
  lcon = [0.0]
  ucon = [0.0]

  return ADNLPModel(A, b, x0, σls=2.0, c=c, lcon=lcon, ucon=ucon)
end

abstract type HS6Types <: AbstractNLPModel end

mutable struct HS6 <: HS6Types # Direct use of obj end
  meta :: NLPModelMeta
  counters :: Counters
end

mutable struct HS6SO <: HS6Types # Use single objectives
  meta :: NLPModelMeta
  σfs :: Array{Float64}
  counters :: Counters
end

mutable struct HS6NLS <: HS6Types # Use Nonlinear Least Squares
  meta :: NLPModelMeta
  σnls :: Float64
  counters :: Counters
end

mutable struct HS6LS <: AbstractLinearLeastSquaresModel # Use Linear Least Squares
  meta :: NLPModelMeta
  A :: Matrix
  b :: Vector
  σls :: Float64
  counters :: Counters
end

function HS6()
  meta = NLPModelMeta(2, nobjs=0, nlsequ=0, llsrows=0, ncon=1,
                      x0=[-1.2; 1.0], lcon=[0.0], ucon=[0.0], name="hs6")

  return HS6(meta, Counters())
end

function HS6SO()
  meta = NLPModelMeta(2, nobjs=1, nlsequ=0, llsrows=0, ncon=1,
                      x0=[-1.2; 1.0], lcon=[0.0], ucon=[0.0], name="hs6")

  return HS6SO(meta, [1.0], Counters())
end

function HS6NLS()
  meta = NLPModelMeta(2, nobjs=0, nlsequ=1, llsrows=0, ncon=1,
                      x0=[-1.2; 1.0], lcon=[0.0], ucon=[0.0], name="hs6")

  return HS6NLS(meta, 2.0, Counters())
end

function HS6LS()
  meta = NLPModelMeta(2, nobjs=0, nlsequ=0, llsrows=1, ncon=1,
                      x0=[-1.2; 1.0], lcon=[0.0], ucon=[0.0], name="hs6")

  A, b = [-1.0 0.0], [-1.0]
  return HS6LS(meta, A, b, 2.0, Counters())
end

function NLPModels.obj(nlp :: HS6, x :: AbstractVector)
  increment!(nlp, :neval_obj)
  return (1 - x[1])^2
end

function NLPModels.grad!(nlp :: HS6, x :: AbstractVector, gx :: AbstractVector)
  increment!(nlp, :neval_grad)
  gx .= [2 * (x[1] - 1); 0.0]
  return gx
end

function NLPModels.hess(nlp :: HS6, x :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hess)
  w = length(y) > 0 ? y[1] : 0.0
  return [2.0 * obj_weight - 20 * w   0.0; 0.0 0.0]
end

function NLPModels.hess_coord(nlp :: HS6, x :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hess)
  w = length(y) > 0 ? y[1] : 0.0
  return ([1], [1], [2.0 * obj_weight - 20 * w])
end

function NLPModels.hprod!(nlp :: HS6, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hprod)
  w = length(y) > 0 ? y[1] : 0.0
  Hv .= [(2.0 * obj_weight - 20 * w) * v[1]; 0.0]
  return Hv
end

function NLPModels.hprod(nlp :: HS6, x :: AbstractVector, v :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hprod)
  w = length(y) > 0 ? y[1] : 0.0
  return [(2.0 * obj_weight - 20 * w) * v[1]; 0.0]
end

function NLPModels.obj(nlp :: HS6SO, i :: Int, x :: AbstractVector)
  @assert i == 1
  increment!(nlp, :neval_iobj)
  return (1 - x[1])^2
end

function NLPModels.grad!(nlp :: HS6SO, i :: Int, x :: AbstractVector, gx :: AbstractVector)
  @assert i == 1
  increment!(nlp, :neval_igrad)
  gx .= [2 * (x[1] - 1); 0.0]
  return gx
end

function NLPModels.grad(nlp :: HS6SO, i :: Int, x :: AbstractVector)
  @assert i == 1
  increment!(nlp, :neval_igrad)
  return [2 * (x[1] - 1); 0.0]
end

function NLPModels.hess(nlp :: HS6SO, i :: Int, x :: AbstractVector)
  @assert i == 1
  increment!(nlp, :neval_ihess)
  return [2.0 0.0; 0.0 0.0]
end

function NLPModels.hess_coord(nlp :: HS6SO, i :: Int, x :: AbstractVector)
  @assert i == 1
  increment!(nlp, :neval_ihess)
  return ([1], [1], [2.0])
end

function NLPModels.hprod!(nlp :: HS6SO, i :: Int, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector)
  @assert i == 1
  increment!(nlp, :neval_ihprod)
  Hv .= [2.0 * v[1]; 0.0]
  return Hv
end

function NLPModels.hprod(nlp :: HS6SO, i :: Int, x :: AbstractVector, v :: AbstractVector)
  @assert i == 1
  Hv = zeros(2)
  return hprod!(nlp, i, x, v, Hv)
end

function NLPModels.residual!(nlp :: HS6NLS, x :: AbstractVector, Fx :: AbstractVector)
  increment!(nlp, :neval_residual)
  Fx .= [1 - x[1]]
  return Fx
end

function NLPModels.residual(nlp :: HS6NLS, x :: AbstractVector)
  increment!(nlp, :neval_residual)
  Fx = [1 - x[1]]
  return Fx
end

function NLPModels.jac_residual(nlp :: HS6NLS, x :: AbstractVector)
  increment!(nlp, :neval_jac_residual)
  return [-1.0 0.0]
end

function NLPModels.jprod_residual!(nlp :: HS6NLS, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jprod_residual)
  Jv .= [-v[1]]
  return Jv
end

function NLPModels.jprod_residual(nlp :: HS6NLS, x :: AbstractVector, v :: AbstractVector)
  increment!(nlp, :neval_jprod_residual)
  return [-v[1]]
end

function NLPModels.jtprod_residual!(nlp :: HS6NLS, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod_residual)
  Jtv .= [-v[1]; 0.0]
  return Jtv
end

function NLPModels.jtprod_residual(nlp :: HS6NLS, x :: AbstractVector, v :: AbstractVector)
  increment!(nlp, :neval_jtprod_residual)
  return [-v[1]; 0.0]
end

function NLPModels.hess_residual(nlp :: HS6NLS, x :: AbstractVector, i :: Int)
  increment!(nlp, :neval_hess_residual)
  return zeros(2,2)
end

function NLPModels.hprod_residual!(nlp :: HS6NLS, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  @assert i == 1
  increment!(nlp, :neval_hess_residual)
  Hiv .= 0.0
  return Hiv
end

function NLPModels.cons!(nlp :: Union{HS6Types,HS6LS}, x :: AbstractVector, cx :: AbstractVector)
  increment!(nlp, :neval_cons)
  cx[1] = 10 * (x[2] - x[1]^2)
  return cx
end

function NLPModels.cons(nlp :: Union{HS6Types,HS6LS}, x :: AbstractVector)
  cx = zeros(1)
  return cons!(nlp, x, cx)
end

function NLPModels.jac(nlp :: Union{HS6Types,HS6LS}, x :: AbstractVector)
  increment!(nlp, :neval_jac)
  return [-20 * x[1]  10.0]
end

function NLPModels.jprod!(nlp :: Union{HS6Types,HS6LS}, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jprod)
  Jv .= [-20 * x[1] * v[1] + 10 * v[2]]
  return Jv
end

function NLPModels.jprod(nlp :: Union{HS6Types,HS6LS}, x :: AbstractVector, v :: AbstractVector)
  increment!(nlp, :neval_jprod)
  return [-20 * x[1] * v[1] + 10 * v[2]]
end

function NLPModels.jtprod!(nlp :: Union{HS6Types,HS6LS}, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod)
  Jtv .= [-20 * x[1]; 10] * v[1]
  return Jtv
end

function NLPModels.jtprod(nlp :: Union{HS6Types,HS6LS}, x :: AbstractVector, v :: AbstractVector)
  increment!(nlp, :neval_jtprod)
  return [-20 * x[1]; 10] * v[1]
end

function NLPModels.jth_hprod(nlp :: Union{HS6Types,HS6LS}, x :: AbstractVector, v :: AbstractVector, i :: Int)
  @assert i == 1
  increment!(nlp, :neval_jhprod)
  return [-20 * v[1]; 0.0]
end

function NLPModels.jth_hprod!(nlp :: Union{HS6Types,HS6LS}, x :: AbstractVector, v :: AbstractVector, i :: Int, Hiv :: AbstractVector)
  @assert i == 1
  increment!(nlp, :neval_jhprod)
  Hiv .= [-20 * v[1]; 0.0]
  return Hiv
end

function NLPModels.chess(nlp :: Union{HS6Types,HS6LS}, i :: Int, x :: AbstractVector)
  @assert i == 1
  increment!(nlp, :neval_ihess)
  return [-20.0 0.0; 0.0 0.0]
end
