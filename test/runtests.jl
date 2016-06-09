using NLPModels
using JuMP
using AmplNLReader
using Base.Test

# A problem with zero variables doesn't make sense.
@test_throws(ErrorException, NLPModelMeta(0))

# SimpleNLPModel with no functions
model = SimpleNLPModel(zeros(2), x->dot(x,x))
for meth in filter(f -> isa(eval(f), Function), names(NLPModels))
  meth = eval(meth)
  @test_throws(NotImplementedError, meth(model))
end

include("genrose.jl")
model = JuMPNLPModel(genrose())
for f in fieldnames(model.counters)
  @assert getfield(model.counters, f) == 0
end

obj(model, model.meta.x0)
@assert model.counters.neval_obj == 1

reset!(model)
@assert model.counters.neval_obj == 0

@test_throws(NotImplementedError, jth_con(model))

include("test_slack_model.jl")

@printf("For tests to pass, the JuMP and AMPL models must have been written identically.\n")
@printf("Constraints, if any, must have been declared in the same order.\n")
@printf("In addition, the AMPL model must have been decoded with preprocessing disabled.\n")
include("consistency.jl")

include("test_mpb.jl")

include("test_simple_model.jl")

