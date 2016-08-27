module NLPModels

using Compat

export AbstractNLPModelMeta, NLPModelMeta, AbstractNLPModel, Counters
export reset!,
       obj, grad, grad!,
       cons, cons!, jth_con, jth_congrad, jth_congrad!, jth_sparse_congrad,
       jac_coord, jac, jprod, jprod!, jtprod, jtprod!,
       jth_hprod, jth_hprod!, ghjvprod, ghjvprod!,
       hess_coord, hess, hprod, hprod!,
       varscale, lagscale, conscale,
       NLPtoMPB, NotImplementedError


include("nlp_utils.jl");
include("nlp_types.jl");

type NotImplementedError <: Exception
  name :: Union{Symbol,Function,ASCIIString}
end

Base.showerror(io::IO, e::NotImplementedError) = print(io, e.name,
  " not implemented")

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
obj(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("obj"))

"""`grad(nlp, x)`

Evaluate \$\\nabla f(x)\$, the gradient of the objective function at `x`.
"""
grad(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("grad"))

"""`grad!(nlp, x, g)`

Evaluate \$\\nabla f(x)\$, the gradient of the objective function at `x` in place.
"""
grad!(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("grad!"))

"""`cons(nlp, x)`

Evaluate \$c(x)\$, the constraints at `x`.
"""
cons(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("cons"))

"""`cons!(nlp, x, c)`

Evaluate \$c(x)\$, the constraints at `x` in place.
"""
cons!(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("cons!"))

jth_con(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("jth_con"))
jth_congrad(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("jth_congrad"))
jth_congrad!(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("jth_congrad!"))
jth_sparse_congrad(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("jth_sparse_congrad"))

"""`(rows,cols,vals) = jac_coord(nlp, x)`

Evaluate \$\\nabla c(x)\$, the constraint's Jacobian at `x` in sparse coordinate format.
"""
jac_coord(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("jac_coord"))

"""`Jx = jac(nlp, x)`

Evaluate \$\\nabla c(x)\$, the constraint's Jacobian at `x` as a sparse matrix.
"""
jac(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("jac"))

"""`Jv = jprod(nlp, x, v)`

Evaluate \$\\nabla c(x)v\$, the Jacobian-vector product at `x`.
"""
jprod(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("jprod"))

"""`Jv = jprod!(nlp, x, v, Jv)`

Evaluate \$\\nabla c(x)v\$, the Jacobian-vector product at `x` in place.
"""
jprod!(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("jprod!"))

"""`Jtv = jtprod(nlp, x, v, Jtv)`

Evaluate \$\\nabla c(x)^Tv\$, the transposed-Jacobian-vector product at `x`.
"""
jtprod(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("jtprod"))

"""`Jtv = jtprod!(nlp, x, v, Jtv)`

Evaluate \$\\nabla c(x)^Tv\$, the transposed-Jacobian-vector product at `x` in place.
"""
jtprod!(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("jtprod!"))

jth_hprod(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("jth_hprod"))
jth_hprod!(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("jth_hprod!"))
ghjvprod(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("ghjvprod"))
ghjvprod!(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("ghjvprod!"))

"""`(rows,cols,vals) = hess_coord(nlp, x; obj_weight=1.0, y=zeros)`

Evaluate the Lagrangian Hessian at `(x,y)` in sparse coordinate format,
with objective function scaled by `obj_weight`, i.e.,

\\\\[ \\nabla^2L(x,y) = \\sigma * \\nabla^2 f(x) + \\sum_{i=1}^m y_i\\nabla^2 c_i(x), \\\\]

with σ = obj_weight.
Only the lower triangle is returned.
"""
hess_coord(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("hess_coord"))

"""`Hx = hess(nlp, x; obj_weight=1.0, y=zeros)`

Evaluate the Lagrangian Hessian at `(x,y)` as a sparse matrix,
with objective function scaled by `obj_weight`, i.e.,

\\\\[ \\nabla^2L(x,y) = \\sigma * \\nabla^2 f(x) + \\sum_{i=1}^m y_i\\nabla^2 c_i(x), \\\\]

with σ = obj_weight.
Only the lower triangle is returned.
"""
hess(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("hess"))

"""`Hv = hprod(nlp, x, v; obj_weight=1.0, y=zeros)`

Evaluate the product of the Lagrangian Hessian at `(x,y)` with the vector `v`,
with objective function scaled by `obj_weight`, i.e.,

\\\\[ \\nabla^2L(x,y) = \\sigma * \\nabla^2 f(x) + \\sum_{i=1}^m y_i\\nabla^2 c_i(x), \\\\]

with σ = obj_weight.
"""
hprod(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("hprod"))

"""`Hv = hprod!(nlp, x, v, Hv; obj_weight=1.0, y=zeros)`

Evaluate the product of the Lagrangian Hessian at `(x,y)` with the vector `v` in
place, with objective function scaled by `obj_weight`, i.e.,

\\\\[ \\nabla^2L(x,y) = \\sigma * \\nabla^2 f(x) + \\sum_{i=1}^m y_i\\nabla^2 c_i(x), \\\\]

with σ = obj_weight.
"""
hprod!(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("hprod!"))

varscale(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("varscale"))
lagscale(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("lagscale"))
conscale(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("conscale"))

"""`mp = NLPtoMPB(nlp, solver)`

Return a `MathProgBase` model corresponding to this model.
`solver` should be a solver instance, e.g., `IpoptSolver()`.
Currently, all models are treated as nonlinear models.
"""
NLPtoMPB(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("NLPtoMPB"))

if Pkg.installed("JuMP") != nothing
  include("jump_model.jl")
end
include("simple_model.jl")

include("slack_model.jl")

end # module
