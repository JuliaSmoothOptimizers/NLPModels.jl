using Test, NLPModels, LinearAlgebra, LinearOperators, Printf, SparseArrays

# Including problems so that they won't be multiply loaded
for problem in [:brownden, :genrose, :hs5, :hs6, :hs10, :hs11, :hs14]
  include("$problem.jl")
end

println("Testing printing of nlp.meta")
print(ADNLPModel(x->0, zeros(10), lvar=[-ones(5); -Inf*ones(5)],
                 uvar=[ones(3); Inf*ones(4); collect(2:4)],
                 name="Unconstrained example").meta)
print(ADNLPModel(x->0, zeros(10), c=x->[0.0;0.0;0.0], lcon=[0.0;0.0;-Inf],
                 ucon=[Inf;0.0;0.0], name="Constrained example").meta)

# A problem with zero variables doesn't make sense.
@test_throws(ErrorException, NLPModelMeta(0))

include("test_not_implemented.jl")
include("test_empty_model.jl")

# ADNLPModel with no functions
model = ADNLPModel(x->dot(x,x), zeros(2), name="square")
@assert model.meta.name == "square"

model = genrose_autodiff()
for counter in fieldnames(typeof(model.counters))
  @eval @assert $counter(model) == 0
end

obj(model, model.meta.x0)
@assert neval_obj(model) == 1

reset!(model)
@assert neval_obj(model) == 0

@test_throws(NotImplementedError, jth_con(model, model.meta.x0, 1))

#=
include("test_tools.jl")

include("test_slack_model.jl")
include("test_qn_model.jl")
=#

@printf("For tests to pass, all models must have been written identically.\n")
@printf("Constraints, if any, must have been declared in the same order.\n")

include("consistency.jl")
@printf("%24s\tConsistency   Derivative Check   Quasi-Newton  Slack variant\n", " ")
#for problem in ["brownden", "hs5", "hs6", "hs10", "hs11", "hs14"]
for problem in ["hs5", "hs6", "hs6so", "hs6nls", "hs6ls"]
  consistency(problem)
end

# Consistency between single objectives, NLS and LS problem
println("Checking consistency of HS6 in single objective, NLS and LS format")
consistent_nlps([HS6(), HS6SO(), HS6NLS(), HS6LS()])

include("test_autodiff_model.jl")
#include("test_view_subarray.jl")
