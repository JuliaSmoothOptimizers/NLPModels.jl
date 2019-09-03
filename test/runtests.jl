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

# Default methods should throw NotImplementedError.
mutable struct DummyModel <: AbstractNLPModel
  meta :: NLPModelMeta
end
model = DummyModel(NLPModelMeta(1))
@test_throws(NotImplementedError, lagscale(model, 1.0))
for meth in [:obj, :varscale, :conscale]
  @eval @test_throws(NotImplementedError, $meth(model, [0]))
end
for meth in [:grad!, :cons!, :jac_structure!, :hess_structure!, :jac_coord!, :hess_coord!]
  @eval @test_throws(NotImplementedError, $meth(model, [0], [1]))
end
for meth in [:jth_con, :jth_congrad, :jth_sparse_congrad]
  @eval @test_throws(NotImplementedError, $meth(model, [0], 1))
end
@test_throws(NotImplementedError, jth_congrad!(model, [0], 1, [2]))
for meth in [:jprod!, :jtprod!, :hprod!, :hess_coord!]
  @eval @test_throws(NotImplementedError, $meth(model, [0], [1], [2]))
end
@test_throws(NotImplementedError, jth_hprod(model, [0], [1], 2))
@test_throws(NotImplementedError, jth_hprod!(model, [0], [1], 2, [3]))
for meth in [:ghjvprod!, :hprod!]
  @eval @test_throws(NotImplementedError, $meth(model, [0], [1], [2], [3]))
end
@assert isa(hess_op(model, [0.]), LinearOperator)
@assert isa(jac_op(model, [0.]), LinearOperator)

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

include("test_tools.jl")

include("test_slack_model.jl")
include("test_qn_model.jl")

@printf("For tests to pass, all models must have been written identically.\n")
@printf("Constraints, if any, must have been declared in the same order.\n")

include("consistency.jl")
@printf("%24s\tConsistency   Derivative Check   Quasi-Newton  Slack variant\n", " ")
for problem in ["brownden", "hs5", "hs6", "hs10", "hs11", "hs14"]
  consistency(problem)
end

include("test_autodiff_model.jl")
include("test_nlsmodels.jl")
include("nls_consistency.jl")
consistent_nls()
include("test_feasibility_form_nls.jl")
include("multiple-precision.jl")
include("test_view_subarray.jl")
test_view_subarrays()
include("test_memory_of_coord.jl")
test_memory_of_coord()
