@testset "Basic Counters check" begin
  nlp = SimpleNLPModel()

  for counter in fieldnames(Counters)
    @eval @test $counter($nlp) == 0
  end

  obj(nlp, nlp.meta.x0)
  grad(nlp, nlp.meta.x0)
  @test sum_counters(nlp) == 2
  reset!(nlp)
  @test sum_counters(nlp) == 0
end