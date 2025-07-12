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

@testset "Counters for jprod and jtprod using jacobian sparsity structure" begin
  nlp = SimpleNLPModel()
  x = nlp.meta.x0
  v = ones(nlp.meta.nvar)
  w = ones(nlp.meta.ncon)
  Jv = similar(w)
  Jtw = similar(v)

  # Test counters for jprod! with sparsity structure
  reset!(nlp)
  initial_jprod_count = neval_jprod(nlp)
  jprod!(nlp, jac_structure(nlp)..., jac_coord(nlp, x), v, Jv)
  @test neval_jprod(nlp) == initial_jprod_count + 1

  # Test counters for jtprod! with sparsity structure
  reset!(nlp)
  initial_jtprod_count = neval_jtprod(nlp)
  jtprod!(nlp, jac_structure(nlp)..., jac_coord(nlp, x), w, Jtw)
  @test neval_jtprod(nlp) == initial_jtprod_count + 1

  # Test counters for jprod_nln! with sparsity structure
  reset!(nlp)
  initial_jprod_nln_count = neval_jprod_nln(nlp)
  jprod_nln!(nlp, jac_nln_structure(nlp)..., jac_nln_coord(nlp, x), v, Jv[2:2])
  @test neval_jprod_nln(nlp) == initial_jprod_nln_count + 1

  # Test counters for jtprod_nln! with sparsity structure
  reset!(nlp)
  initial_jtprod_nln_count = neval_jtprod_nln(nlp)
  jtprod_nln!(nlp, jac_nln_structure(nlp)..., jac_nln_coord(nlp, x), w[2:2], Jtw)
  @test neval_jtprod_nln(nlp) == initial_jtprod_nln_count + 1

  # Test counters for jprod_lin! with sparsity structure
  reset!(nlp)
  initial_jprod_lin_count = neval_jprod_lin(nlp)
  jprod_lin!(nlp, jac_lin_structure(nlp)..., jac_lin_coord(nlp, x), v, Jv[1:1])
  @test neval_jprod_lin(nlp) == initial_jprod_lin_count + 1

  # Test counters for jtprod_lin! with sparsity structure
  reset!(nlp)
  initial_jtprod_lin_count = neval_jtprod_lin(nlp)
  jtprod_lin!(nlp, jac_lin_structure(nlp)..., jac_lin_coord(nlp, x), w[1:1], Jtw)
  @test neval_jtprod_lin(nlp) == initial_jtprod_lin_count + 1
end

if VERSION ≥ VersionNumber(1, 7, 3)
  @testset "Allocations for NLP counters" begin
    nlp = SimpleNLPModel()

    increment!(nlp, :neval_obj)
    alloc_mem = @allocated increment!(nlp, :neval_obj)
    @test alloc_mem == 0
  end
end
