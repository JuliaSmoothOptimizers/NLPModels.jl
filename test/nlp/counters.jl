@testset "Basic Counters check" begin
  nlp = SimpleNLPModel()

  for counter in fieldnames(Counters)
    @eval @test $counter($nlp) == 0
  end

  obj(nlp, nlp.meta.x0)
  grad(nlp, nlp.meta.x0)
  @test sum_counters(nlp) == 2

  for counter in fieldnames(Counters)
    increment!(nlp, counter)
  end
  # sum all counters of problem `nlp` except 
  # `cons`, `jac`, `jprod` and `jtprod` = 20-4+2
  @test sum_counters(nlp) == 18

  reset!(nlp)
  @test sum_counters(nlp) == 0
end

if VERSION ≥ VersionNumber(1, 7, 3)
  @testset "Allocations for NLP counters" begin
    nlp = SimpleNLPModel()
    
    increment!(nlp, :neval_obj)
    alloc_mem = @allocated increment!(nlp, :neval_obj)
    @test alloc_mem == 0
  end
end