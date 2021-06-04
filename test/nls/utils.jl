mutable struct SuperNLSModel{T, S} <: AbstractNLSModel{T, S}
  model
end

@testset "Increase coverage of default_nlscounters" begin
  @default_nlscounters SuperNLSModel model
  nls = SuperNLSModel{Float64, Vector{Float64}}(SimpleNLSModel())
  increment!(nls, :neval_residual)
  @test neval_residual(nls.model) == 1
end
