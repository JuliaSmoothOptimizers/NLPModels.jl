using ForwardDiff

export ADNLPModel, obj, grad, grad!, cons, cons!, jac_coord, jac, jprod,
       jprod!, jtprod, jtprod!, hess, hprod, hprod!,
       residual!, jac_residual, jprod_residual!, jtprod_residual!,
       jac_op_residual, hess_residual, hprod_residual!, cons, cons!, jac_coord,
       jac, jprod, jprod!, jtprod, jtprod!, hess, hess_coord, hprod, hprod!

"""ADNLPModel is an AbstractNLPModel using ForwardDiff to compute the
derivatives.
In this interface, the objective function \$f\$ and an initial estimate are
required. If there are constraints, the function
\$c:\\mathbb{R}^n\\rightarrow\\mathbb{R}^m\$  and the vectors
\$c_L\$ and \$c_U\$ also need to be passed. Bounds on the variables and an
inital estimate to the Lagrangian multipliers can also be provided.

````
ADNLPModel(f, x0; lvar = [-∞,…,-∞], uvar = [∞,…,∞], y0 = zeros,
  c = NotImplemented, lcon = [-∞,…,-∞], ucon = [∞,…,∞], name = "Generic")
````

  - `f :: Function` - The objective function \$f\$;
  - `x0 :: AbstractVector` - The initial point of the problem;
  - `lvar :: AbstractVector` - \$\\ell\$, the lower bound of the variables;
  - `uvar :: AbstractVector` - \$u\$, the upper bound of the variables;
  - `c :: Function` - The constraints function \$c\$;
  - `y0 :: AbstractVector` - The initial value of the Lagrangian estimates;
  - `lcon :: AbstractVector` - \$c_L\$, the lower bounds of the constraints function;
  - `ucon :: AbstractVector` - \$c_U\$, the upper bounds of the constraints function;
  - `name :: String` - A name for the model.

The functions follow the same restrictions of ForwardDiff functions, summarised
here:

  - The function can only be composed of generic Julia functions;
  - The function must accept only one argument;
  - The function's argument must accept a subtype of AbstractVector;
  - The function should be type-stable.

For contrained problems, the function \$c\$ is required, and it must return
an array even when m = 1,
and \$c_L\$ and \$c_U\$ should be passed, otherwise the problem is ill-formed.
For equality constraints, the corresponding index of \$c_L\$ and \$c_U\$ should be the
same.
"""
mutable struct ADNLPModel <: AbstractNLPModel
  meta :: NLPModelMeta

  counters :: Counters

  # Functions
  fs   :: Array{Function}
  σfs  :: Array{Float64}
  F    :: Function
  σnls :: Float64
  A    :: Union{AbstractMatrix,AbstractLinearOperator}
  b    :: AbstractVector
  σls  :: Float64
  c    :: Function
end

acceptsObjective(::ADNLPModel) = true
acceptsNLS(::ADNLPModel) = true
acceptsLS(::ADNLPModel) = true
acceptsMultipleObjective(::ADNLPModel) = true

function ADNLPModel(fs::Array{Function}, σfs::Vector,
                    F::Function, nlsequ::Int, σnls::Float64,
                    A::Union{AbstractMatrix,AbstractLinearOperator}, b::AbstractVector, σls::Float64,
                    x0::AbstractVector; y0::AbstractVector = Float64[],
                    lvar::AbstractVector = Float64[], uvar::AbstractVector = Float64[],
                    lcon::AbstractVector = Float64[], ucon::AbstractVector = Float64[],
                    c::Function = (args...)->throw(NotImplementedError("cons")),
                    name::String = "Generic", lin::AbstractVector{Int}=Int[])

  nvar = length(x0)
  length(lvar) == 0 && (lvar = -Inf*ones(nvar))
  length(uvar) == 0 && (uvar =  Inf*ones(nvar))
  ncon = maximum([length(lcon); length(ucon); length(y0)])

  nobjs = length(fs)
  @assert nobjs == length(σfs)
  llsrows = size(A,1)
  @assert llsrows == length(b)
  @assert all(σfs .!= 0.0)
  @assert nlsequ == 0 || σnls > 0
  @assert llsrows == 0 || σls > 0

  # TODO: Figure out best way to compute nnzh
  nnzh = nvar * (nvar + 1) / 2
  nnzj = 0

  if ncon > 0
    length(lcon) == 0 && (lcon = -Inf*ones(ncon))
    length(ucon) == 0 && (ucon =  Inf*ones(ncon))
    length(y0) == 0   && (y0 = zeros(ncon))
    nnzj = nvar * ncon
  end
  nln = setdiff(1:ncon, lin)

  meta = NLPModelMeta(nvar, nobjs=nobjs, nlsequ=nlsequ, llsrows=llsrows,
                      x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, y0=y0,
                      lcon=lcon, ucon=ucon, nnzj=nnzj, nnzh=nnzh,
                      lin=lin, nln=nln, minimize=true, islp=false,
                      name=name)

  return ADNLPModel(meta, Counters(), fs, σfs, F, σnls, A, b, σls, c)
end

ADNLPModel(f::Function, x0::AbstractVector; kwargs...) =
  ADNLPModel(Function[f], [1.0], (x)->NotImplementedError, 0, 0.0,
             zeros(0,length(x0)), zeros(0), 0.0, x0; kwargs...)

ADNLPModel(fs::Array{Function}, σfs::Array, x0::AbstractVector; kwargs...) =
  ADNLPModel(fs, σfs, (x)->NotImplementedError, 0, 0.0,
             zeros(0,length(x0)), zeros(0), 0.0, x0; kwargs...)

ADNLPModel(F::Function, nequ::Int, x0::AbstractVector; kwargs...) =
  ADNLPModel(Function[], zeros(0), F, nequ, 1.0,
             zeros(0,length(x0)), zeros(0), 0.0, x0; kwargs...)

ADNLPModel(A::Union{AbstractMatrix,AbstractLinearOperator}, b::AbstractVector, x0::AbstractVector; kwargs...) =
  ADNLPModel(Function[], zeros(0), (x)->NotImplementedError, 0, 0.0,
             A, b, 1.0, x0; kwargs...)

# API implementation

function obj(nlp :: ADNLPModel, i :: Int, x :: AbstractVector)
  increment!(nlp, :neval_iobj)
  return nlp.fs[i](x)
end

function residual!(nlp :: ADNLPModel, x :: AbstractVector, Fx :: AbstractVector)
  increment!(nlp, :neval_residual)
  Fx[:] = nlp.F(x)
  return Fx
end

function grad(nlp :: ADNLPModel, i :: Int, x :: AbstractVector)
  increment!(nlp, :neval_igrad)
  return ForwardDiff.gradient(nlp.fs[i], x)
end

function grad!(nlp :: ADNLPModel, i :: Int, x :: AbstractVector, g :: AbstractVector)
  increment!(nlp, :neval_igrad)
  return ForwardDiff.gradient!(view(g, 1:length(x)), nlp.fs[i], x)
end

function hess_coord(nlp :: ADNLPModel, i :: Int, x :: AbstractVector)
  return findnz(hess(nlp, x))
end

function hess(nlp :: ADNLPModel, i :: Int, x :: AbstractVector)
  increment!(nlp, :neval_ihess)
  return tril(ForwardDiff.hessian(nlp.fs[i], x))
end

function hprod(nlp :: ADNLPModel, i :: Int, x :: AbstractVector, v :: AbstractVector)
  increment!(nlp, :neval_ihprod)
  return ForwardDiff.hessian(nlp.fs[i], x) * v
end

function hprod!(nlp :: ADNLPModel, i :: Int, x :: AbstractVector, v :: AbstractVector,
                Hv :: AbstractVector)
  increment!(nlp, :neval_ihprod)
  Hv[1:nvar(nlp)] = ForwardDiff.hessian(nlp.fs[i], x) * v
  return Hv
end

function chess(nlp :: ADNLPModel, i :: Int, x :: AbstractVector)
  increment!(nlp, :neval_jhess)
  return tril(ForwardDiff.hessian(x->nlp.c(x)[i], x))
end

function jth_con(nlp :: ADNLPModel, i :: Int, x :: AbstractVector)
  increment!(nlp, :neval_jcon)
  return nlp.c(x)[i]
end

function jth_hprod(nlp :: ADNLPModel, x :: AbstractVector, v :: AbstractVector, i :: Int)
  increment!(nlp, :neval_jhprod)
  return ForwardDiff.hessian(x->nlp.c(x)[i], x) * v
end

function jth_hprod!(nlp :: ADNLPModel, x :: AbstractVector, v :: AbstractVector, i :: Int, Hv :: AbstractVector)
  increment!(nlp, :neval_jhprod)
  Hv[1:nvar(nlp)] = ForwardDiff.hessian(x->nlp.c(x)[i], x) * v
  return Hv
end

function cons(nlp :: ADNLPModel, x :: AbstractVector)
  increment!(nlp, :neval_cons)
  return nlp.c(x)
end

function cons!(nlp :: ADNLPModel, x :: AbstractVector, c :: AbstractVector)
  increment!(nlp, :neval_cons)
  c[1:nlp.meta.ncon] = nlp.c(x)
  return c
end

function jac_coord(nlp :: ADNLPModel, x :: AbstractVector)
  increment!(nlp, :neval_jac)
  J = ForwardDiff.jacobian(nlp.c, x)
  rows = [j for i = 1:nlp.meta.nvar for j = 1:nlp.meta.ncon]
  cols = [i for i = 1:nlp.meta.nvar for j = 1:nlp.meta.ncon]
  vals = [J[j,i] for i = 1:nlp.meta.nvar for j = 1:nlp.meta.ncon]
  return rows, cols, vals
end

function jac(nlp :: ADNLPModel, x :: AbstractVector)
  increment!(nlp, :neval_jac)
  return ForwardDiff.jacobian(nlp.c, x)
end

function jprod(nlp :: ADNLPModel, x :: AbstractVector, v :: AbstractVector)
  increment!(nlp, :neval_jprod)
  return ForwardDiff.jacobian(nlp.c, x) * v
end

function jprod!(nlp :: ADNLPModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jprod)
  Jv[1:nlp.meta.ncon] = ForwardDiff.jacobian(nlp.c, x) * v
  return Jv
end

function jtprod(nlp :: ADNLPModel, x :: AbstractVector, v :: AbstractVector)
  increment!(nlp, :neval_jtprod)
  return ForwardDiff.jacobian(nlp.c, x)' * v
end

function jtprod!(nlp :: ADNLPModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod)
  Jtv[1:nlp.meta.nvar] = ForwardDiff.jacobian(nlp.c, x)' * v
  return Jtv
end

function jac_residual(nlp :: ADNLPModel, x :: AbstractVector)
  increment!(nlp, :neval_jac_residual)
  return ForwardDiff.jacobian(nlp.F, x)
end

function jprod_residual!(nlp :: ADNLPModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jprod_residual)
  Jv[:] = ForwardDiff.jacobian(nlp.F, x) * v
  return Jv
end

function jtprod_residual!(nlp :: ADNLPModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod_residual)
  Jtv[:] = ForwardDiff.jacobian(nlp.F, x)' * v
  return Jtv
end

function hess_residual(nlp :: ADNLPModel, x :: AbstractVector, i :: Int)
  increment!(nlp, :neval_hess_residual)
  return tril(ForwardDiff.hessian(x->nlp.F(x)[i], x))
end

function hprod_residual!(nlp :: ADNLPModel, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  increment!(nlp, :neval_hprod_residual)
  Hiv[:] = ForwardDiff.hessian(x->nlp.F(x)[i], x) * v
  return Hiv
end

