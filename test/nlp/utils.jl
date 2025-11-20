mutable struct SuperNLPModel{T, S} <: AbstractNLPModel{T, S}
  model
end

@testset "Testing @lencheck, @lencheck_tup, @rangecheck" begin
  x = zeros(2)
  @lencheck 2 x
  @test_throws DimensionError @lencheck 1 x
  @test_throws DimensionError @lencheck 3 x

  xs = (zeros(2), ones(2))
  @lencheck_tup 2 xs
  xs_bad = (zeros(2), ones(3))
  err = try
    @lencheck_tup 2 xs_bad
    nothing
  catch e
    e
  end
  @test isa(err, DimensionError)
  @test err.name == "xs_bad[2]"
  @test err.dim_expected == 2
  @test err.dim_found == 3

  @rangecheck 1 3 2
  @test_throws ErrorException @rangecheck 1 3 0
  @test_throws ErrorException @rangecheck 1 3 4

  io = IOBuffer()
  showerror(io, DimensionError(:A, 1, 2))
  @test String(take!(io)) == "DimensionError: Input A should have length 1 not 2"
end

@testset "Increase coverage of default_NLPcounters" begin
  @default_counters SuperNLPModel model
  nlp = SuperNLPModel{Float64, Vector{Float64}}(SimpleNLPModel())
  increment!(nlp, :neval_obj)
  @test neval_obj(nlp.model) == 1
  @test nlp.counters == nlp.model.counters
end
