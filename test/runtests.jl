using Base.Test, Compat, Ipopt, JuMP, MathProgBase, NLPModels, LinearOperators

# Including problems so that they won't be multiply loaded
for problem in [:brownden, :genrose, :hs5, :hs6, :hs10, :hs11, :hs14, :hs15]
  include("$problem.jl")
end

# A problem with zero variables doesn't make sense.
@test_throws(ErrorException, NLPModelMeta(0))

# Default methods should throw NotImplementedError.
type DummyModel <: AbstractNLPModel
  meta :: NLPModelMeta
end
model = DummyModel(NLPModelMeta(1))
@test_throws(NotImplementedError, lagscale(model, 1.0))
for meth in [:obj, :grad, :cons,  :jac, :jac_coord, :hess, :hess_coord, :varscale, :conscale]
  @eval @test_throws(NotImplementedError, $meth(model, [0]))
end
for meth in [:grad!, :cons!, :jprod, :jtprod, :hprod]
  @eval @test_throws(NotImplementedError, $meth(model, [0], [1]))
end
for meth in [:jth_con, :jth_congrad, :jth_sparse_congrad]
  @eval @test_throws(NotImplementedError, $meth(model, [0], 1))
end
@test_throws(NotImplementedError, jth_congrad!(model, [0], 1, [2]))
for meth in [:jprod!, :jtprod!, :hprod!, :ghjvprod]
  @eval @test_throws(NotImplementedError, $meth(model, [0], [1], [2]))
end
@test_throws(NotImplementedError, jth_hprod(model, [0], [1], 2))
@test_throws(NotImplementedError, jth_hprod!(model, [0], [1], 2, [3]))
@test_throws(NotImplementedError, ghjvprod!(model, [0], [1], [2], [3]))
@assert isa(hess_op(model, [0.]), LinearOperator)

# ADNLPModel with no functions
model = ADNLPModel(x->dot(x,x), zeros(2), name="square")
ignore_throw = [:reset!, :hess_op, :gradient_check, :hessian_check,
    :jacobian_check, :hessian_check_from_grad]
@assert model.meta.name == "square"

model = MathProgNLPModel(genrose(), name="genrose")
@assert model.meta.name == "genrose"
for counter in fieldnames(model.counters)
  @eval @assert $counter(model) == 0
end

obj(model, model.meta.x0)
@assert neval_obj(model) == 1

reset!(model)
@assert neval_obj(model) == 0

@test_throws(NotImplementedError, jth_con(model, model.meta.x0, 1))

include("test_slack_model.jl")

@printf("For tests to pass, all models must have been written identically.\n")
@printf("Constraints, if any, must have been declared in the same order.\n")

include("consistency.jl")
@printf("%24s\tConsistency   Derivative Check   Slack variant\n", " ")
for problem in [:brownden, :hs5, :hs6, :hs10, :hs11, :hs14]
  consistency(problem)
end

include("test_mpb.jl")

include("test_autodiff_model.jl")
include("test_simple_model.jl")
