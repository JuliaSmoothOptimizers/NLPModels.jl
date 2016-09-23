using Compat
using JuMP
using NLPModels

using Base.Test

# Including problems so that they won't be multiply loaded
for problem in [:brownden, :genrose, :hs5, :hs6, :hs10, :hs11, :hs14, :hs15]
  include("$problem.jl")
end

# A problem with zero variables doesn't make sense.
@test_throws(ErrorException, NLPModelMeta(0))

# ADNLPModel with no functions
model = ADNLPModel(x->dot(x,x), zeros(2), name="square")
@assert model.meta.name == "square"
for meth in filter(f -> isa(eval(f), Function), names(NLPModels))
  meth in (:reset!, :hess_op) && continue
  meth in fieldnames(model.counters) && continue
  meth = eval(meth)
  @test_throws(NotImplementedError, meth(model))
end

model = JuMPNLPModel(genrose(), name="genrose")
@assert model.meta.name == "genrose"
for counter in fieldnames(model.counters)
  @eval @assert $counter(model) == 0
end

obj(model, model.meta.x0)
@assert neval_obj(model) == 1

reset!(model)
@assert neval_obj(model) == 0

@test_throws(NotImplementedError, jth_con(model))

include("test_slack_model.jl")

@printf("For tests to pass, all models must have been written identically.\n")
@printf("Constraints, if any, must have been declared in the same order.\n")
include("consistency.jl")

include("test_mpb.jl")

include("test_autodiff_model.jl")
include("test_simple_model.jl")

