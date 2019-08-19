mutable struct DummyNLSModel <: AbstractNLSModel
end

model = DummyNLSModel()

@eval @test_throws(NotImplementedError, jac_residual(model, [0]))
for mtd in [:residual!, :hess_residual]
  @eval @test_throws(NotImplementedError, $mtd(model, [0], [1]))
end
for mtd in [:jprod_residual!, :jtprod_residual!]
  @eval @test_throws(NotImplementedError, $mtd(model, [0], [1], [2]))
end
@test_throws(NotImplementedError, jth_hess_residual(model, [0], 1))
@test_throws(NotImplementedError, hprod_residual!(model, [0], 1, [2], [3]))
