@testset "Increase coverage of NLSCounters" begin
  nls = SimpleNLSModel()
  obj(nls, nls.meta.x0)
  residual(nls, nls.meta.x0)
  jac_residual(nls, nls.meta.x0)
  @test neval_obj(nls) == 1
  @test neval_residual(nls) == 2
  @test neval_jac_residual(nls) == 1
  @test sum_counters(nls) == 4
  reset!(nls)
  @test sum_counters(nls) == 0
end
