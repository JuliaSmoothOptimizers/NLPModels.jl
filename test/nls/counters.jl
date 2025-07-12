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

@testset "Counters for jprod and jtprod using jacobian sparsity structure" begin
  nls = SimpleNLSModel()
  x = nls.meta.x0
  v = ones(nls.meta.nvar)
  w = ones(nls.meta.ncon)
  Jv = similar(w)
  Jtw = similar(v)

  # Test counters for jprod! with sparsity structure
  reset!(nls)
  initial_jprod_count = neval_jprod(nls)
  jprod!(nls, jac_structure(nls)..., jac_coord(nls, x), v, Jv)
  @test neval_jprod(nls) == initial_jprod_count + 1

  # Test counters for jtprod! with sparsity structure
  reset!(nls)
  initial_jtprod_count = neval_jtprod(nls)
  jtprod!(nls, jac_structure(nls)..., jac_coord(nls, x), w, Jtw)
  @test neval_jtprod(nls) == initial_jtprod_count + 1

  # Test counters for jprod_residual! with sparsity structure
  w_residual = ones(nls.nls_meta.nequ)
  Jv_residual = similar(w_residual)
  reset!(nls)
  initial_jprod_residual_count = neval_jprod_residual(nls)
  jprod_residual!(nls, jac_structure_residual(nls)..., jac_coord_residual(nls, x), v, Jv_residual)
  @test neval_jprod_residual(nls) == initial_jprod_residual_count + 1

  # Test counters for jtprod_residual! with sparsity structure
  reset!(nls)
  initial_jtprod_residual_count = neval_jtprod_residual(nls)
  jtprod_residual!(nls, jac_structure_residual(nls)..., jac_coord_residual(nls, x), w_residual, Jtw)
  @test neval_jtprod_residual(nls) == initial_jtprod_residual_count + 1
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
