mutable struct SuperNLSModel <: AbstractNLSModel
  model
end

@testset "Increase coverage of default_nlscounters" begin
  @default_nlscounters SuperNLSModel model
  nls = SuperNLSModel(SimpleNLSModel())
  increment!(nls, :neval_residual)
  @test neval_residual(nls.model) == 1
end
