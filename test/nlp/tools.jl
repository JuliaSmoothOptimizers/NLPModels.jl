@testset "Problem type functions" begin
  foo_list = [
    has_bounds,
    bound_constrained,
    unconstrained,
    linearly_constrained,
    equality_constrained,
    inequality_constrained,
    has_equalities,
    has_inequalities,
  ]
  meta_list = [
    NLPModelMeta(2),
    NLPModelMeta(2; lvar=zeros(2), uvar=ones(2)),
    NLPModelMeta(2; ncon=1, lcon=[0.0], ucon=[0.0]),
    NLPModelMeta(2; ncon=1, lcon=[0.0], ucon=[1.0]),
    NLPModelMeta(2; ncon=1, lcon=[0.0], ucon=[Inf]),
    NLPModelMeta(2; ncon=1, lcon=[-Inf], ucon=[0.0]),
    NLPModelMeta(2; ncon=1, lcon=[0.0], ucon=[1.0], lin=[1]),
    NLPModelMeta(2; ncon=2, lcon=[0.0, 0.0], ucon=[1.0, 1.0], lin=[1]),
    NLPModelMeta(2; ncon=2, lcon=[0.0, 0.0], ucon=[1.0, 0.0], lin=[1]),
    NLPModelMeta(2; lvar=zeros(2), uvar=ones(2), ncon=1, lcon=[0.0], ucon=[0.0]),
    NLPModelMeta(2; lvar=zeros(2), uvar=ones(2), ncon=1, lcon=[0.0], ucon=[1.0]),
    NLPModelMeta(2; lvar=zeros(2), uvar=ones(2), ncon=1, lcon=[0.0], ucon=[Inf]),
    NLPModelMeta(2; lvar=zeros(2), uvar=ones(2), ncon=1, lcon=[-Inf], ucon=[0.0]),
    NLPModelMeta(2; lvar=zeros(2), uvar=ones(2), ncon=1, lcon=[0.0], ucon=[1.0], lin=[1]),
    NLPModelMeta(
      2; lvar=zeros(2), uvar=ones(2), ncon=2, lcon=[0.0, 0.0], ucon=[1.0, 1.0], lin=[1]
    ),
    NLPModelMeta(
      2; lvar=zeros(2), uvar=ones(2), ncon=2, lcon=[0.0, 0.0], ucon=[1.0, 0.0], lin=[1]
    ),
  ]
  results = Bool[
    0 1 0 0 0 0 0 0 0 1 1 1 1 1 1 1
    0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0
    0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0
    0 0 0 1 1 1 1 1 0 0 1 1 1 1 1 0
    0 0 1 0 0 0 0 0 1 1 0 0 0 0 0 1
    0 0 0 1 1 1 1 1 1 0 1 1 1 1 1 1
  ]
  for (i, f) in enumerate(foo_list), (j, meta) in enumerate(meta_list)
    @test f(meta) == results[i, j]
    @test f(DummyModel(meta)) == results[i, j]
  end
end
