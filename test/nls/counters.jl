@testset "Increase coverage of NLSCounters" begin

  nls = SimpleNLSModel()

  obj(nls, nls.meta.x0)
  residual(nls, nls.meta.x0)
  jac_residual(nls, nls.meta.x0)

  @test neval_obj(nls) == 1
  @test neval_residual(nls) == 2
  @test neval_jac_residual(nls) == 1
  @test sum_counters(nls) == 4

  for counter in fieldnames(Counters)
    increment!(nls, counter)
  end

  for counter in fieldnames(NLSCounters)
    counter == :counters && continue
    increment!(nls, counter)
  end

  # sum all counters of problem `nlp` except 
  # `cons`, `jac`, `jprod` and `jtprod` = 20+7-4+4
  @test sum_counters(nls) == 27

  reset!(nls)
  @test sum_counters(nls) == 0
end

if VERSION ≥ VersionNumber(1, 7, 3)
  @testset "Allocations for NLS counters" begin
    nls = SimpleNLSModel()

    increment!(nls, :neval_obj)
    alloc_mem = @allocated increment!(nls, :neval_obj)
    @test alloc_mem == 0
    
    increment!(nls, :neval_residual)
    alloc_mem2 = @allocated increment!(nls, :neval_residual)
    @test alloc_mem2 == 0
  end
end