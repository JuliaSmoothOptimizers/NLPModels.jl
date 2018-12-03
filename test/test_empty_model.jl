mutable struct DummyEmptyModel <: AbstractNLPModel
  meta     :: NLPModelMeta
  counters :: Counters
end

function test_empty_model()
  n = 3
  nlp = DummyEmptyModel(NLPModelMeta(n), Counters())
  x = nlp.meta.x0
  @test obj(nlp, x) == 0.0
  @test grad(nlp, x) == zeros(n)
  @test hess(nlp, x) == zeros(n, n)
  @test hess_op(nlp, x) isa LinearOperator
  @test jac_op(nlp, x) isa LinearOperator
end

test_empty_model()
