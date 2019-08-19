using Test, NLPModels, LinearAlgebra, LinearOperators, Printf, SparseArrays

const problems = ["BROWNDEN", "HS5", "HS6", "HS10", "HS11", "HS14"]

# Including problems so that they won't be multiply loaded
for problem in Symbol.(lowercase.(problems))
  include("$problem.jl")
end

println("Testing printing of nlp.meta")
print(NLPModelMeta(10, x0=zeros(10), lvar=[-ones(5); -Inf*ones(5)],
                   uvar=[ones(3); Inf*ones(4); collect(2:4)],
                   name="Unconstrained example"))
print(NLPModelMeta(10, x0=zeros(10), ncon=3, lcon=[0.0;0.0;-Inf],
                   ucon=[Inf;0.0;0.0], name="Constrained example"))

# A problem with zero variables doesn't make sense.
@test_throws(ErrorException, NLPModelMeta(0))

# Default methods should throw NotImplementedError.
mutable struct DummyModel <: AbstractNLPModel
  meta :: NLPModelMeta
end
dummy = DummyModel(NLPModelMeta(1))
@test_throws(NotImplementedError, lagscale(dummy, 1.0))
for meth in [:obj, :varscale, :conscale]
  @eval @test_throws(NotImplementedError, $meth(dummy, [0]))
end
for meth in [:grad!, :cons!, :jac_structure!, :hess_structure!]
  @eval @test_throws(NotImplementedError, $meth(dummy, [0], [1]))
end
for meth in [:jth_con, :jth_congrad, :jth_sparse_congrad]
  @eval @test_throws(NotImplementedError, $meth(dummy, [0], 1))
end
@test_throws(NotImplementedError, jth_congrad!(dummy, [0], 1, [2]))
for meth in [:jprod!, :jtprod!, :hprod!]
  @eval @test_throws(NotImplementedError, $meth(dummy, [0], [1], [2]))
end
@test_throws(NotImplementedError, jth_hprod(dummy, [0], [1], 2))
@test_throws(NotImplementedError, jth_hprod!(dummy, [0], [1], 2, [3]))
for meth in [:jac_coord!, :hess_coord!, :ghjvprod!]
  @eval @test_throws(NotImplementedError, $meth(dummy, [0], [1], [2], [3]))
end
@test_throws(NotImplementedError, jth_con(dummy, dummy.meta.x0, 1))

@test isa(hess_op(dummy, [0.]), LinearOperator)
@test isa(jac_op(dummy, [0.]), LinearOperator)

for p in problems
  model = eval(Symbol(p))()
  for counter in fieldnames(typeof(model.counters))
    @eval @test $counter($model) == 0
  end

  obj(model, model.meta.x0)
  @test neval_obj(model) == 1

  reset!(model)
  @test neval_obj(model) == 0
end

include("test_tools.jl")

include("test_slack_model.jl")
include("test_qn_model.jl")

@printf("For tests to pass, all models must have been written identically.\n")
@printf("Constraints, if any, must have been declared in the same order.\n")

include("consistency.jl")
@printf("%24s\tConsistency   Derivative Check   Quasi-Newton  Slack variant\n", " ")
for problem in ["brownden", "hs5", "hs6", "hs10", "hs11", "hs14"]
  @printf("Checking problem %-20s", problem)
  consistent_nlps([eval(Meta.parse(uppercase(problem)))()])
end

include("test_view_subarray.jl")
test_view_subarrays()
include("test_nlsmodels.jl")
include("nls_consistency.jl")
for problem in ["hs6nls"]
  @printf("Checking problem %-20s", problem)
  consistent_nlss([eval(Meta.parse(uppercase(problem)))()])
end
include("multiple-precision.jl")
include("test_memory_of_coord.jl")
test_memory_of_coord()
