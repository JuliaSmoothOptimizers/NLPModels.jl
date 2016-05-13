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
       NLPtoMPB


include("nlp_utils.jl");
include("nlp_types.jl");


"Reset evaluation counters"
function reset!(counters :: Counters)
  for f in fieldnames(Counters)
    setfield!(counters, f, 0)
  end
  return counters
end

# Methods to be overridden in other packages.
reset!(nlp :: AbstractNLPModel) = error("reset!() not implemented")
obj(nlp :: AbstractNLPModel, args...; kwargs...) = error("obj() not implemented")
grad(nlp :: AbstractNLPModel, args...; kwargs...) = error("grad() not implemented")
grad!(nlp :: AbstractNLPModel, args...; kwargs...) = error("grad!() not implemented")

cons(nlp :: AbstractNLPModel, args...; kwargs...) = error("cons() not implemented")
cons!(nlp :: AbstractNLPModel, args...; kwargs...) = error("cons!() not implemented")

jth_con(nlp :: AbstractNLPModel, args...; kwargs...) = error("jth_con() not implemented")
jth_congrad(nlp :: AbstractNLPModel, args...; kwargs...) = error("jth_congrad() not implemented")
jth_congrad!(nlp :: AbstractNLPModel, args...; kwargs...) = error("jth_congrad!() not implemented")
jth_sparse_congrad(nlp :: AbstractNLPModel, args...; kwargs...) = error("jth_sparse_congrad() not implemented")

jac_coord(nlp :: AbstractNLPModel, args...; kwargs...) = error("jac_coord() not implemented")
jac(nlp :: AbstractNLPModel, args...; kwargs...) = error("jac() not implemented")
jprod(nlp :: AbstractNLPModel, args...; kwargs...) = error("jprod() not implemented")
jprod!(nlp :: AbstractNLPModel, args...; kwargs...) = error("jprod!() not implemented")
jtprod(nlp :: AbstractNLPModel, args...; kwargs...) = error("jtprod() not implemented")
jtprod!(nlp :: AbstractNLPModel, args...; kwargs...) = error("jtprod!() not implemented")

jth_hprod(nlp :: AbstractNLPModel, args...; kwargs...) = error("jth_hprod() not implemented")
jth_hprod!(nlp :: AbstractNLPModel, args...; kwargs...) = error("jth_hprod!() not implemented")
ghjvprod(nlp :: AbstractNLPModel, args...; kwargs...) = error("ghjvprod() not implemented")
ghjvprod!(nlp :: AbstractNLPModel, args...; kwargs...) = error("ghjvprod!() not implemented")

hess_coord(nlp :: AbstractNLPModel, args...; kwargs...) = error("hess_coord() not implemented")
hess(nlp :: AbstractNLPModel, args...; kwargs...) = error("hess() not implemented")
hprod(nlp :: AbstractNLPModel, args...; kwargs...) = error("hprod() not implemented")
hprod!(nlp :: AbstractNLPModel, args...; kwargs...) = error("hprod!() not implemented")

varscale(nlp :: AbstractNLPModel, args...; kwargs...) = error("varscale() not implemented")
lagscale(nlp :: AbstractNLPModel, args...; kwargs...) = error("lagscale() not implemented")
conscale(nlp :: AbstractNLPModel, args...; kwargs...) = error("conscale() not implemented")

NLPtoMPB(nlp :: AbstractNLPModel, args...; kwargs...) = error("NLPtoMPB() not implemented")

if Pkg.installed("JuMP") != nothing
  include("jump_model.jl")
end

include("slack_model.jl")

end # module
