@testset "Problem type functions" begin
  @testset "Analysis = $bool" for bool in (false, true)
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
      NLPModelMeta(2, variable_bounds_analysis = bool, constraint_bounds_analysis = bool),
      NLPModelMeta(
        2,
        lvar = zeros(2),
        uvar = ones(2),
        variable_bounds_analysis = bool,
        constraint_bounds_analysis = bool,
      ),
      NLPModelMeta(
        2,
        ncon = 1,
        lcon = [0.0],
        ucon = [0.0],
        variable_bounds_analysis = bool,
        constraint_bounds_analysis = bool,
      ),
      NLPModelMeta(
        2,
        ncon = 1,
        lcon = [0.0],
        ucon = [1.0],
        variable_bounds_analysis = bool,
        constraint_bounds_analysis = bool,
      ),
      NLPModelMeta(
        2,
        ncon = 1,
        lcon = [0.0],
        ucon = [Inf],
        variable_bounds_analysis = bool,
        constraint_bounds_analysis = bool,
      ),
      NLPModelMeta(
        2,
        ncon = 1,
        lcon = [-Inf],
        ucon = [0.0],
        variable_bounds_analysis = bool,
        constraint_bounds_analysis = bool,
      ),
      NLPModelMeta(
        2,
        ncon = 1,
        lcon = [0.0],
        ucon = [1.0],
        lin = [1],
        variable_bounds_analysis = bool,
        constraint_bounds_analysis = bool,
      ),
      NLPModelMeta(
        2,
        ncon = 2,
        lcon = [0.0, 0.0],
        ucon = [1.0, 1.0],
        lin = [1],
        variable_bounds_analysis = bool,
        constraint_bounds_analysis = bool,
      ),
      NLPModelMeta(
        2,
        ncon = 2,
        lcon = [0.0, 0.0],
        ucon = [1.0, 0.0],
        lin = [1],
        variable_bounds_analysis = bool,
        constraint_bounds_analysis = bool,
      ),
      NLPModelMeta(
        2,
        lvar = zeros(2),
        uvar = ones(2),
        ncon = 1,
        lcon = [0.0],
        ucon = [0.0],
        variable_bounds_analysis = bool,
        constraint_bounds_analysis = bool,
      ),
      NLPModelMeta(
        2,
        lvar = zeros(2),
        uvar = ones(2),
        ncon = 1,
        lcon = [0.0],
        ucon = [1.0],
        variable_bounds_analysis = bool,
        constraint_bounds_analysis = bool,
      ),
      NLPModelMeta(
        2,
        lvar = zeros(2),
        uvar = ones(2),
        ncon = 1,
        lcon = [0.0],
        ucon = [Inf],
        variable_bounds_analysis = bool,
        constraint_bounds_analysis = bool,
      ),
      NLPModelMeta(
        2,
        lvar = zeros(2),
        uvar = ones(2),
        ncon = 1,
        lcon = [-Inf],
        ucon = [0.0],
        variable_bounds_analysis = bool,
        constraint_bounds_analysis = bool,
      ),
      NLPModelMeta(
        2,
        lvar = zeros(2),
        uvar = ones(2),
        ncon = 1,
        lcon = [0.0],
        ucon = [1.0],
        lin = [1],
        variable_bounds_analysis = bool,
        constraint_bounds_analysis = bool,
      ),
      NLPModelMeta(
        2,
        lvar = zeros(2),
        uvar = ones(2),
        ncon = 2,
        lcon = [0.0, 0.0],
        ucon = [1.0, 1.0],
        lin = [1],
        variable_bounds_analysis = bool,
        constraint_bounds_analysis = bool,
      ),
      NLPModelMeta(
        2,
        lvar = zeros(2),
        uvar = ones(2),
        ncon = 2,
        lcon = [0.0, 0.0],
        ucon = [1.0, 0.0],
        lin = [1],
        variable_bounds_analysis = bool,
        constraint_bounds_analysis = bool,
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
    for f in fieldnames(NLPModelMeta), (j, meta) in enumerate(meta_list)
      @test eval(Meta.parse("get_" * string(f)))(meta) == getproperty(meta, f)
      @test eval(Meta.parse("get_" * string(f)))(DummyModel(meta)) == getproperty(meta, f)
    end
  end
end
