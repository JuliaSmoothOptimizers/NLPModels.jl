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

@testset "Basic increment of NLSCounters" begin
  nls = SimpleNLSModel()
  increment_neval_obj!(nls)
  increment_neval_residual!(nls)
  increment_neval_jac_residual!(nls)
  @test neval_obj(nls) == 1
  @test neval_residual(nls) == 1
  @test neval_jac_residual(nls) == 1
  @test sum_counters(nls) == 3
  reset!(nls)
  @test sum_counters(nls) == 0
end
