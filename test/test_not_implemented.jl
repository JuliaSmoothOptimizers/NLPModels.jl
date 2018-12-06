mutable struct DummyModelNotImplemented <: AbstractNLPModel
  meta     :: NLPModelMeta
  counters :: Counters
end

function test_not_implemented()
  # Default methods should throw NotImplementedError.
  model = DummyModelNotImplemented(NLPModelMeta(1), Counters())
  @test_throws(NotImplementedError, lagscale(model, 1.0))

  global model
  for meth in [:cons,  :jac, :jac_coord, :varscale, :conscale, :residual]
    @eval @test_throws(NotImplementedError, $meth(model, [0.0]))
  end
  for meth in [:cons!, :jprod, :jtprod, :residual!, :jprod_residual, :jtprod_residual]
    @eval @test_throws(NotImplementedError, $meth(model, [0.0], [1.0]))
  end
  for meth in [:obj, :grad, :hess, :hess_coord, :chess, :jth_con, :jth_congrad, :jth_sparse_congrad, :hess_residual]
    @eval @test_throws(NotImplementedError, $meth(model, 0, [1.0]))
  end
  for meth in [:jprod!, :jtprod!, :ghjvprod, :jprod_residual!, :jtprod_residual!]
    @eval @test_throws(NotImplementedError, $meth(model, [0.0], [1.0], [2.0]))
  end
  for meth in [:grad!, :hprod, :jth_hprod, :jth_congrad!, :hprod_residual]
    @eval @test_throws(NotImplementedError, $meth(model, 0, [1.0], [2.0]))
  end
  @test_throws(NotImplementedError, ghjvprod!(model, [0.0], [1.0], [2.0], [3.0]))
  for meth in [:hprod!, :hprod_residual!, :jth_hprod!]
    @eval @test_throws(NotImplementedError, $meth(model, 0, [1.0], [2.0], [3.0]))
  end
end

test_not_implemented()
