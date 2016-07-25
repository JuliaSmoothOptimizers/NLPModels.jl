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

"Reset evaluation counters"
function reset!(counters :: Counters)
  for f in fieldnames(Counters)
    setfield!(counters, f, 0)
  end
  return counters
end

"Reset evaluation count in `nlp`"
function reset!(nlp :: AbstractNLPModel)
  reset!(nlp.counters)
  return nlp
end

# Methods to be overridden in other packages.
obj(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("obj"))
grad(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("grad"))
grad!(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("grad!"))

cons(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("cons"))
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

jac_coord(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("jac_coord"))
jac(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("jac"))
jprod(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("jprod"))
jprod!(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("jprod!"))
jtprod(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("jtprod"))
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

hess_coord(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("hess_coord"))
hess(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("hess"))
hprod(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("hprod"))
hprod!(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("hprod!"))

varscale(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("varscale"))
lagscale(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("lagscale"))
conscale(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("conscale"))

NLPtoMPB(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("NLPtoMPB"))

if Pkg.installed("JuMP") != nothing
  include("jump_model.jl")
end
include("simple_model.jl")

include("slack_model.jl")

end # module
