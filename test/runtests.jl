include("TestUtils/TestUtils.jl")
using .TestUtils

using Test, NLPModels, LinearAlgebra, LinearOperators, Printf, SparseArrays

@info("Testing printing of nlp.meta")
print(IOBuffer(), ADNLPModel(x->0, zeros(10), [-ones(5); -Inf*ones(5)],
                 [ones(3); Inf*ones(4); collect(2:4)],
                 name="Unconstrained example").meta)
print(IOBuffer(), ADNLPModel(x->0, zeros(10), x->[0.0;0.0;0.0], [0.0;0.0;-Inf],
                 [Inf;0.0;0.0], name="Constrained example").meta)

# A problem with zero variables doesn't make sense.
@test_throws(ErrorException, NLPModelMeta(0))

# Default methods should throw MethodError since they're not defined
mutable struct DummyModel <: AbstractNLPModel
  meta :: NLPModelMeta
end
model = DummyModel(NLPModelMeta(1))
@test_throws(MethodError, lagscale(model, 1.0))
for meth in [:obj, :varscale, :conscale]
  @eval @test_throws(MethodError, $meth(model, [0.0]))
end
for meth in [:jac_structure!, :hess_structure!]
  @eval @test_throws(MethodError, $meth(model, [0], [1]))
end
for meth in [:grad!, :cons!, :jac_coord!]
  @eval @test_throws(MethodError, $meth(model, [0.0], [1.0]))
end
for meth in [:jth_con, :jth_congrad, :jth_sparse_congrad]
  @eval @test_throws(MethodError, $meth(model, [0.0], 1))
end
@test_throws(MethodError, jth_congrad!(model, [0.0], 1, [2.0]))
for meth in [:jprod!, :jtprod!]
  @eval @test_throws(MethodError, $meth(model, [0.0], [1.0], [2.0]))
end
@test_throws(MethodError, jth_hprod(model, [0.0], [1.0], 2))
@test_throws(MethodError, jth_hprod!(model, [0.0], [1.0], 2, [3.0]))
for meth in [:ghjvprod!]
  @eval @test_throws(MethodError, $meth(model, [0.0], [1.0], [2.0], [3.0]))
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

@test_throws(MethodError, jth_con(model, model.meta.x0, 1))
include("test_tools.jl")

include("test_slack_model.jl")
include("test_qn_model.jl")
include("nlp_testutils.jl")
include("nls_testutils.jl")
include("test_autodiff_model.jl")
include("test_nlsmodels.jl")
include("test_feasibility_form_nls.jl")