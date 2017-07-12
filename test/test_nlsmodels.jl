type DummyNLSModel <: AbstractNLSModel
end

model = DummyNLSModel()

for mtd in [:residual, :jac_residual]
  @eval @test_throws(NotImplementedError, $mtd(model, [0]))
end
for mtd in [:residual!, :jprod_residual, :jtprod_residual]
  @eval @test_throws(NotImplementedError, $mtd(model, [0], [1]))
end
for mtd in [:jprod_residual!, :jtprod_residual!]
  @eval @test_throws(NotImplementedError, $mtd(model, [0], [1], [2]))
end
@test_throws(NotImplementedError, hess_residual(model, [0], 1))
@test_throws(NotImplementedError, hprod_residual(model, [0], 1, [2]))
@test_throws(NotImplementedError, hprod_residual!(model, [0], 1, [2], [3]))

include("test_autodiff_nls_model.jl")
include("test_lls_model.jl")
include("test_simple_nls_model.jl")
include("test_feasibility_nls_model.jl")
