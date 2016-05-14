using NLPModels
using JuMP
using AmplNLReader
using Base.Test

# A problem with zero variables doesn't make sense.
@test_throws(ErrorException, NLPModelMeta(0))

type DummyNLPModel <: AbstractNLPModel
end

# Initially, no method is implemented.
model = DummyNLPModel()
for meth in filter(f -> isa(eval(f), Function), names(NLPModels))
  meth = eval(meth)
  @test_throws(ErrorException, meth(model))
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

@test_throws(ErrorException, jth_con(model))

include("test_slack_model.jl")

@printf("For tests to pass, the JuMP and AMPL models must have been written identically.\n")
@printf("Constraints, if any, must have been declared in the same order.\n")
@printf("In addition, the AMPL model must have been decoded with preprocessing disabled.\n")
include("jump_vs_ampl.jl")

include("test_mpb.jl")
