mutable struct SuperNLPModel{T, S} <: AbstractNLPModel{T, S}
  model
end

@testset "Testing @lencheck and @rangecheck" begin
  x = zeros(2)
  @lencheck 2 x
  @test_throws DimensionError @lencheck 1 x
  @test_throws DimensionError @lencheck 3 x

  @rangecheck 1 3 2
  @test_throws ErrorException @rangecheck 1 3 0
  @test_throws ErrorException @rangecheck 1 3 4

  io = IOBuffer()
  showerror(io, DimensionError(:A, 1, 2))
  @test String(take!(io)) == "DimensionError: Input A should have length 1 not 2"
end

@testset "Increase coverage of default_NLPcounters" begin
  @default_counters SuperNLPModel model (neval_hprod,)
  nlp = SuperNLPModel{Float64, Vector{Float64}}(SimpleNLPModel())
  increment!(nlp, :neval_obj)
  @test neval_obj(nlp.model) == 1
  @test get_counters(nlp) == get_counters(nlp.model)
  @test neval_hprod(nlp) == 0  # because counters are forwarded, even though neval_hprod has not been forwarded
end
