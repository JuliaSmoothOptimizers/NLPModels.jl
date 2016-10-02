module NLPModels

using Compat
import Compat.String

using LinearOperators

export AbstractNLPModelMeta, NLPModelMeta, AbstractNLPModel, Counters
export reset!,
       obj, grad, grad!,
       cons, cons!, jth_con, jth_congrad, jth_congrad!, jth_sparse_congrad,
       jac_coord, jac, jprod, jprod!, jtprod, jtprod!,
       jth_hprod, jth_hprod!, ghjvprod, ghjvprod!,
       hess_coord, hess, hprod, hprod!, hess_op,
       push!,
       varscale, lagscale, conscale,
       NotImplementedError

# import methods we override
import Base.push!
import LinearOperators.reset!

include("nlp_utils.jl");
include("nlp_types.jl");

type NotImplementedError <: Exception
  name :: Union{Symbol,Function,String}
end

Base.showerror(io::IO, e::NotImplementedError) = print(io, e.name, " not implemented")


# simple default API for retrieving counters
for counter in fieldnames(Counters)
  @eval begin
    $counter(nlp :: AbstractNLPModel) = nlp.counters.$counter
    export $counter
  end
end

"""`reset!(counters)`

Reset evaluation counters
"""
function reset!(counters :: Counters)
  for f in fieldnames(Counters)
    setfield!(counters, f, 0)
  end
  return counters
end

"""`reset!(nlp)

Reset evaluation count in `nlp`
"""
function reset!(nlp :: AbstractNLPModel)
  reset!(nlp.counters)
  return nlp
end

# Methods to be overridden in other packages.
"""`obj(nlp, x)`

Evaluate \$f(x)\$, the objective function of `nlp` at `x`.
"""
obj(::AbstractNLPModel, ::AbstractVector) =
  throw(NotImplementedError("obj"))

"""`grad(nlp, x)`

Evaluate \$\\nabla f(x)\$, the gradient of the objective function at `x`.
"""
grad(::AbstractNLPModel, ::AbstractVector) =
  throw(NotImplementedError("grad"))

"""`grad!(nlp, x, g)`

Evaluate \$\\nabla f(x)\$, the gradient of the objective function at `x` in place.
"""
grad!(::AbstractNLPModel, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("grad!"))

"""`cons(nlp, x)`

Evaluate \$c(x)\$, the constraints at `x`.
"""
cons(::AbstractNLPModel, ::AbstractVector) =
  throw(NotImplementedError("cons"))

"""`cons!(nlp, x, c)`

Evaluate \$c(x)\$, the constraints at `x` in place.
"""
cons!(::AbstractNLPModel, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("cons!"))

jth_con(::AbstractNLPModel, ::AbstractVector, ::Integer) =
  throw(NotImplementedError("jth_con"))
jth_congrad(::AbstractNLPModel, ::AbstractVector, ::Integer) =
  throw(NotImplementedError("jth_congrad"))
jth_congrad!(::AbstractNLPModel, ::AbstractVector, ::Integer, ::AbstractVector) =
  throw(NotImplementedError("jth_congrad!"))
jth_sparse_congrad(::AbstractNLPModel, ::AbstractVector, ::Integer) =
  throw(NotImplementedError("jth_sparse_congrad"))

"""`(rows,cols,vals) = jac_coord(nlp, x)`

Evaluate \$\\nabla c(x)\$, the constraint's Jacobian at `x` in sparse coordinate format.
"""
jac_coord(::AbstractNLPModel, ::AbstractVector) =
  throw(NotImplementedError("jac_coord"))

"""`Jx = jac(nlp, x)`

Evaluate \$\\nabla c(x)\$, the constraint's Jacobian at `x` as a sparse matrix.
"""
jac(::AbstractNLPModel, ::AbstractVector) = throw(NotImplementedError("jac"))

"""`Jv = jprod(nlp, x, v)`

Evaluate \$\\nabla c(x)v\$, the Jacobian-vector product at `x`.
"""
jprod(::AbstractNLPModel, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("jprod"))

"""`Jv = jprod!(nlp, x, v, Jv)`

Evaluate \$\\nabla c(x)v\$, the Jacobian-vector product at `x` in place.
"""
jprod!(::AbstractNLPModel, ::AbstractVector, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("jprod!"))

"""`Jtv = jtprod(nlp, x, v, Jtv)`

Evaluate \$\\nabla c(x)^Tv\$, the transposed-Jacobian-vector product at `x`.
"""
jtprod(::AbstractNLPModel, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("jtprod"))

"""`Jtv = jtprod!(nlp, x, v, Jtv)`

Evaluate \$\\nabla c(x)^Tv\$, the transposed-Jacobian-vector product at `x` in place.
"""
jtprod!(::AbstractNLPModel, ::AbstractVector, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("jtprod!"))

jth_hprod(::AbstractNLPModel, ::AbstractVector, ::AbstractVector, ::Integer) =
  throw(NotImplementedError("jth_hprod"))
jth_hprod!(::AbstractNLPModel, ::AbstractVector, ::AbstractVector, ::Integer, ::AbstractVector) =
  throw(NotImplementedError("jth_hprod!"))
ghjvprod(::AbstractNLPModel, ::AbstractVector, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("ghjvprod"))
ghjvprod!(::AbstractNLPModel, ::AbstractVector, ::AbstractVector, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("ghjvprod!"))

"""`(rows,cols,vals) = hess_coord(nlp, x; obj_weight=1.0, y=zeros)`

Evaluate the Lagrangian Hessian at `(x,y)` in sparse coordinate format,
with objective function scaled by `obj_weight`, i.e.,

\\\\[ \\nabla^2L(x,y) = \\sigma * \\nabla^2 f(x) + \\sum_{i=1}^m y_i\\nabla^2 c_i(x), \\\\]

with σ = obj_weight.
Only the lower triangle is returned.
"""
hess_coord(::AbstractNLPModel, ::AbstractVector; kwargs...) =
  throw(NotImplementedError("hess_coord"))

"""`Hx = hess(nlp, x; obj_weight=1.0, y=zeros)`

Evaluate the Lagrangian Hessian at `(x,y)` as a sparse matrix,
with objective function scaled by `obj_weight`, i.e.,

\\\\[ \\nabla^2L(x,y) = \\sigma * \\nabla^2 f(x) + \\sum_{i=1}^m y_i\\nabla^2 c_i(x), \\\\]

with σ = obj_weight.
Only the lower triangle is returned.
"""
hess(::AbstractNLPModel, ::AbstractVector; kwargs...) =
  throw(NotImplementedError("hess"))

"""`Hv = hprod(nlp, x, v; obj_weight=1.0, y=zeros)`

Evaluate the product of the Lagrangian Hessian at `(x,y)` with the vector `v`,
with objective function scaled by `obj_weight`, i.e.,

\\\\[ \\nabla^2L(x,y) = \\sigma * \\nabla^2 f(x) + \\sum_{i=1}^m y_i\\nabla^2 c_i(x), \\\\]

with σ = obj_weight.
"""
hprod(::AbstractNLPModel, ::AbstractVector, ::AbstractVector; kwargs...) =
  throw(NotImplementedError("hprod"))

"""`Hv = hprod!(nlp, x, v, Hv; obj_weight=1.0, y=zeros)`

Evaluate the product of the Lagrangian Hessian at `(x,y)` with the vector `v` in
place, with objective function scaled by `obj_weight`, i.e.,

\\\\[ \\nabla^2L(x,y) = \\sigma * \\nabla^2 f(x) + \\sum_{i=1}^m y_i\\nabla^2 c_i(x), \\\\]

with σ = obj_weight.
"""
hprod!(::AbstractNLPModel, ::AbstractVector, ::AbstractVector, ::AbstractVector; kwargs...) =
  throw(NotImplementedError("hprod!"))

"""`H = hess_op(nlp, x; obj_weight=1.0, y=zeros)`

Return the Lagrangian Hessian at `(x,y)` with objective function scaled by
`obj_weight` as a linear operator. The resulting object may be used as if it were a
matrix, e.g., `H * v`. The linear operator H represents

\\\\[ \\nabla^2L(x,y) = \\sigma * \\nabla^2 f(x) + \\sum_{i=1}^m y_i\\nabla^2 c_i(x), \\\\]

with σ = obj_weight.
"""
function hess_op(nlp :: AbstractNLPModel, x :: Vector{Float64};
                 obj_weight :: Float64=1.0, y :: Vector{Float64}=zeros(nlp.meta.ncon))
  return LinearOperator(nlp.meta.nvar, nlp.meta.nvar,
                        true, true,
                        v -> hprod(nlp, x, v; obj_weight=obj_weight, y=y))
end

push!(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("push!"))
varscale(::AbstractNLPModel, ::AbstractVector) =
  throw(NotImplementedError("varscale"))
lagscale(::AbstractNLPModel, ::Float64) =
  throw(NotImplementedError("lagscale"))
conscale(::AbstractNLPModel, ::AbstractVector) =
  throw(NotImplementedError("conscale"))

if Pkg.installed("MathProgBase") != nothing
  include("mpb_model.jl")
  include("nlp_to_mpb.jl")
  if Pkg.installed("JuMP") != nothing
    include("jump_model.jl")
  end
end
if Pkg.installed("ForwardDiff") != nothing
  include("autodiff_model.jl")
end
include("simple_model.jl")
include("slack_model.jl")
include("qn_model.jl")

include("dercheck.jl")

end # module
