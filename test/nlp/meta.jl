@testset "Sanity checks" begin
  @test_throws ErrorException NLPModelMeta(0)
  @test_throws ErrorException NLPModelMeta(2; nparam = -1)
  @test_throws ErrorException NLPModelMeta(2; nnzjp = -1)
  @test_throws ErrorException NLPModelMeta(2; nnzhp = -1)
  @test_throws ErrorException NLPModelMeta(2; nnzgp = -1)
  @test_throws ErrorException NLPModelMeta(2; nnzjplcon = -1)
  @test_throws ErrorException NLPModelMeta(2; nnzjpucon = -1)
  @test_throws ErrorException NLPModelMeta(2; nnzjplvar = -1)
  @test_throws ErrorException NLPModelMeta(2; nnzjpuvar = -1)
end

@testset "Meta copier." begin
  nlp = SimpleNLPModel()

  # Check simple copy
  meta = NLPModelMeta(nlp.meta)
  for field in fieldnames(typeof(nlp.meta))
    @test getfield(nlp.meta, field) == getfield(meta, field)
  end

  modif = Dict(
    :nnzh => 1,
    :x0 => [2.0; 2.0; 0.0],
    :nvar => 3,
    :lvar => zeros(3),
    :uvar => [1.0; 1.0; 0.0],
  )
  meta = NLPModelMeta(nlp.meta; modif...)

  for field in setdiff(fieldnames(typeof(nlp.meta)), union(keys(modif), [:ifix]))
    @test getfield(nlp.meta, field) == getfield(meta, field)
  end
  for field in keys(modif)
    @test getfield(meta, field) == modif[field]
  end
  @test meta.ifix == [3]
end
