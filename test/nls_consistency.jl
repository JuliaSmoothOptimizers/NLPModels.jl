import LinearAlgebra: I

function consistent_nls_counters(nlss)
  N = length(nlss)
  V = zeros(Int, N)
  for field in fieldnames(NLSCounters)
    field == :counters && continue
    V = [eval(field)(nls) for nls in nlss]
    @test all(V .== V[1])
  end
  V = [sum_counters(nls) for nls in nlss]
  @test all(V .== V[1])
end

function consistent_nls_functions(nlss; rtol=1.0e-8, exclude=[])
  N = length(nlss)
  n = nls_meta(nlss[1]).nvar
  m = nls_meta(nlss[1]).nequ

  tmp_n = zeros(n)
  tmp_m = zeros(m)

  x = 10 * [-(-1.0)^i for i = 1:n]

  if !(residual in exclude)
    Fs = Any[residual(nls, x) for nls in nlss]
    for i = 1:N
      for j = i+1:N
        @test isapprox(Fs[i], Fs[j], rtol=rtol)
      end

      r = residual!(nlss[i], x, tmp_m)
      @test isapprox(r, Fs[i], rtol=rtol)
      @test isapprox(Fs[i], tmp_m, rtol=rtol)
    end
  end

  if intersect([jac_residual,jac_coord_residual], exclude) == []
    Js = Any[jac_residual(nls, x) for nls in nlss]
    for i = 1:N
      for j = i+1:N
        @test isapprox(Js[i], Js[j], rtol=rtol)
      end
      I, J, V = jac_coord_residual(nlss[i], x)
      @test length(I) == length(J) == length(V) == nlss[i].nls_meta.nnzj
      I2, J2 = jac_structure_residual(nlss[i])
      @test I == I2
      @test J == J2
      jac_structure_residual!(nlss[i], I2, J2)
      @test I == I2
      @test J == J2
      tmp_V = zeros(nlss[i].nls_meta.nnzj)
      jac_coord_residual!(nlss[i], x, I, J, tmp_V)
      @test tmp_V == V
    end
  end

  if intersect([jac_op_residual, jprod_residual, jtprod_residual],  exclude) == []
    J_ops = Any[jac_op_residual(nls, x) for nls in nlss]
    Jv, Jtv = zeros(m), zeros(n)
    J_ops_inplace = Any[jac_op_residual!(nls, x, Jv, Jtv) for nls in nlss]

    v = [-(-1.0)^i for i = 1:n]

    Jps = Any[jprod_residual(nls, x, v) for nls in nlss]
    for i = 1:N
      for j = i+1:N
        @test isapprox(Jps[i], Jps[j], rtol=rtol)
      end

      jps = jprod_residual!(nlss[i], x, v, tmp_m)
      @test isapprox(jps, Jps[i], rtol=rtol)
      @test isapprox(Jps[i], tmp_m, rtol=rtol)
      @test isapprox(Jps[i], J_ops[i] * v, rtol=rtol)
      @test isapprox(Jps[i], J_ops_inplace[i] * v, rtol=rtol)
    end

    v = [-(-1.0)^i for i = 1:m]

    Jtps = Any[jtprod_residual(nls, x, v) for nls in nlss]
    for i = 1:N
      for j = i+1:N
        @test isapprox(Jtps[i], Jtps[j], rtol=rtol)
      end

      jtps = jtprod_residual!(nlss[i], x, v, tmp_n)
      @test isapprox(jtps, Jtps[i], rtol=rtol)
      @test isapprox(Jtps[i], tmp_n, rtol=rtol)
      @test isapprox(Jtps[i], J_ops[i]' * v, rtol=rtol)
      @test isapprox(Jtps[i], J_ops_inplace[i]' * v, rtol=rtol)
    end
  end

  if intersect([hess_residual, hprod_residual, hess_op_residual], exclude) == []
    v = [-(-1.0)^i for i = 1:n]
    w = [-(-1.0)^i for i = 1:m]

    Hs = Any[hess_residual(nls, x, w) for nls in nlss]
    Hsi = Any[sum(jth_hess_residual(nls, x, i) * w[i] for i = 1:m) for nls in nlss]
    for i = 1:N
      for j = i+1:N
        @test isapprox(Hs[i], Hs[j], rtol=rtol)
      end
      @test isapprox(Hs[i], Hsi[i], rtol=rtol)
      if !(hess_coord_residual in exclude)
        I, J, V = hess_coord_residual(nlss[i], x, w)
        @test length(I) == length(J) == length(V) == nlss[i].nls_meta.nnzh
        @test sparse(I, J, V, n, n) == Hs[i]
        I2, J2 = hess_structure_residual(nlss[i])
        @test I == I2
        @test J == J2
        hess_structure_residual!(nlss[i], I2, J2)
        @test I == I2
        @test J == J2
        tmp_V = zeros(nlss[i].nls_meta.nnzh)
        hess_coord_residual!(nlss[i], x, w, I, J, tmp_V)
        @test tmp_V == V
      end
    end

    for k = 1:m
      Hs = Any[jth_hess_residual(nls, x, k) for nls in nlss]
      Hvs = Any[hprod_residual(nls, x, k, v) for nls in nlss]
      Hops = Any[hess_op_residual(nls, x, k) for nls in nlss]
      Hiv = zeros(n)
      Hops_inplace = Any[hess_op_residual!(nls, x, k, Hiv) for nls in nlss]
      for i = 1:N
        for j = i+1:N
          @test isapprox(Hs[i], Hs[j], rtol=rtol)
          @test isapprox(Hvs[i], Hvs[j], rtol=rtol)
        end

        hvs = hprod_residual!(nlss[i], x, k, v, tmp_n)
        @test isapprox(hvs, Hvs[i], rtol=rtol)
        @test isapprox(Hvs[i], tmp_n, rtol=rtol)
        @test isapprox(Hvs[i], Hops[i] * v, rtol=rtol)
        @test isapprox(Hvs[i], Hops_inplace[i] * v, rtol=rtol)
      end
    end
  end
end

function consistent_nls()
  @testset "Consistency of Linear problem" begin
    m, n = 50, 20
    A = Matrix(1.0I, m, n) .+ 1
    b = collect(1:m)
    lvar = -ones(n)
    uvar = ones(n)
    lls_model = LLSModel(A, b, lvar=lvar, uvar=uvar)
    autodiff_model = ADNLSModel(x->A*x-b, zeros(n), m, lvar=lvar, uvar=uvar)
    nlp = ADNLPModel(x->0, zeros(n), lvar=lvar, uvar=uvar, c=x->A*x-b,
                     lcon=zeros(m), ucon=zeros(m))
    feas_res_model = FeasibilityResidual(nlp)
    nlss = [lls_model, autodiff_model, feas_res_model]
    consistent_nls_counters(nlss)
    consistent_counters(nlss)
    consistent_nls_functions(nlss)
    consistent_nls_counters(nlss)
    consistent_counters(nlss)
    for nls in nlss
      reset!(nls)
    end

    f(x) = begin
      r = A*x - b
      return 0.5*dot(r, r)
    end
    nlps = [nlss; ADNLPModel(f, zeros(n))]
    consistent_functions(nlps, exclude=[hess_coord])
  end

  @testset "Consistency of Linear problem with linear constraints" begin
    m, n = 50, 20
    A = Matrix(1.0I, m, n) .+ 1
    b = collect(1:m)
    lvar = -ones(n)
    uvar = ones(n)
    nc = 10
    C = [ones(nc, n); 2 * ones(nc, n); -ones(nc, n); -Matrix(1.0I, nc, n)]
    lcon = [   zeros(nc); -ones(nc); fill(-Inf,nc); zeros(nc)]
    ucon = [fill(Inf,nc);  ones(nc);     zeros(nc); zeros(nc)]
    K = ((1:4:4nc) .+ (0:3)')[:]
    lcon, ucon = lcon[K], ucon[K]
    lls_model = LLSModel(A, b, lvar=lvar, uvar=uvar, C=C, lcon=lcon,
                         ucon=ucon)
    autodiff_model = ADNLSModel(x->A*x-b, zeros(n), m, lvar=lvar,
                                uvar=uvar, c=x->C*x, lcon=lcon,
                                ucon=ucon)
    nlss = [lls_model, autodiff_model]
    consistent_nls_counters(nlss)
    consistent_counters(nlss)
    consistent_nls_functions(nlss)
    consistent_nls_counters(nlss)
    consistent_counters(nlss)
    consistent_functions(nlss, exclude=[hess_coord])
  end

  @testset "Consistency of Nonlinear problem" begin
    m, n = 10, 2
    lvar = -ones(n)
    uvar =  ones(n)
    F(x) = [2 + 2i - exp(i*x[1]) - exp(i*x[2]) for i = 1:m]
    x0 = [0.3; 0.4]

    autodiff_model = ADNLSModel(F, x0, m, lvar=lvar, uvar=uvar)
    nlp = ADNLPModel(x->0, x0, lvar=lvar, uvar=uvar, c=F, lcon=zeros(m), ucon=zeros(m))
    feas_res_model = FeasibilityResidual(nlp)
    nlss = [autodiff_model, feas_res_model]
    consistent_nls_counters(nlss)
    consistent_counters(nlss)
    consistent_nls_functions(nlss)
    consistent_nls_counters(nlss)
    consistent_counters(nlss)
    for nls in nlss
      reset!(nls)
    end

    f(x) = begin
      r = F(x)
      return 0.5*dot(r, r)
    end
    nlps = [nlss; ADNLPModel(f, zeros(n))]
    consistent_functions(nlps, exclude=[hess_coord])
  end

  @testset "Consistency of Nonlinear problem with constraints" begin
    m, n = 10, 2
    lvar = -ones(n)
    uvar =  ones(n)
    F(x) = [2 + 2i - exp(i*x[1]) - exp(i*x[2]) for i = 1:m]
    x0 = [0.3; 0.4]
    c(x) = [x[1]^2 - x[2]^2; 2 * x[1] * x[2]; x[1] + x[2]]
    lcon = [0.0; -1.0; -Inf]
    ucon = [Inf;  1.0;  0.0]

    autodiff_model = ADNLSModel(F, x0, m, lvar=lvar, uvar=uvar,
                                lcon=lcon, ucon=ucon, c=c)
    nlss = [autodiff_model]
    consistent_nls_counters(nlss)
    consistent_counters(nlss)
    consistent_nls_functions(nlss)
    consistent_nls_counters(nlss)
    consistent_counters(nlss)
    consistent_functions(nlss, exclude=[hess_coord])
  end

  @testset "Consistency of slack variant" begin
    F(x) = [x[1] - 1.0;
            x[2] - x[1]^2;
            sin(x[1] * x[2]) * x[3]]
    x0 = [0.3; 0.4; 0.5]
    c(x) = [x[1]^2 - x[2]^2;
            2 * x[1] * x[2];
            x[1] + x[2];
            cos(x[1]) - x[2]]
    lcon = [0.0; -1.0; -Inf; 0.0]
    ucon = [Inf;  1.0;  0.0; 0.0]
    nls = ADNLSModel(F, x0, 3, c=c, lcon=lcon, ucon=ucon)

    nls_manual_slack = ADNLSModel(F, [x0; zeros(3)], 3,
                                  c=x->c(x) - [x[4]; x[6]; x[5]; 0.0],
                                  lcon=zeros(4), ucon=zeros(4),
                                  lvar=[-Inf; -Inf; -Inf; 0.0; -Inf; -1.0],
                                  uvar=[Inf; Inf; Inf; Inf; 0.0; 1.0])

    nls_auto_slack = SlackNLSModel(nls)
    nlss = [nls_manual_slack, nls_auto_slack]
    consistent_nls_counters(nlss)
    consistent_counters(nlss)
    consistent_nls_functions(nlss)
    consistent_nls_counters(nlss)
    consistent_counters(nlss)
    consistent_functions(nlss)
  end

  @testset "Consistency of slack variant for linear problem" begin
    A = [1.0 0.0; 1.0 2.0; 2.0 1.0]
    b = [1.0; 3.0; 2.0]
    C = [1.0 1.0; 1.0 -1.0; 2.0 1.0; 1.0 2.0]
    x0 = ones(2)
    lcon = [0.0; -1.0; -Inf; 0.0]
    ucon = [Inf;  1.0;  0.0; 0.0]
    adnls = ADNLSModel(x -> A*x - b, x0, 3, c=x -> C*x, lcon=lcon, ucon=ucon)
    lls = LLSModel(A, b, x0=x0, C=C, lcon=lcon, ucon=ucon)

    nlss = [SlackNLSModel(adnls), SlackNLSModel(lls)]
    consistent_nls_counters(nlss)
    consistent_counters(nlss)
    consistent_nls_functions(nlss)
    consistent_nls_counters(nlss)
    consistent_counters(nlss)
    consistent_functions(nlss, exclude=[hess_coord])
  end

  @testset "Consistency of LLS with Matrix and LinearOperator" begin
    m, n = 50, 20
    A = Matrix(1.0I, m, n) .+ 1
    b = collect(1:m)
    lls = LLSModel(A, b)
    lls2 = LLSModel(LinearOperator(A), b)
    nlss = [lls, lls2]
    consistent_nls_counters(nlss)
    consistent_counters(nlss)
    consistent_nls_functions(nlss, exclude=[jac_residual])
    consistent_nls_counters(nlss)
    consistent_counters(nlss)

    nc = 10
    C = [ones(nc, n); 2 * ones(nc, n); -ones(nc, n); -Matrix(1.0I, nc, n)]
    lcon = [   zeros(nc); -ones(nc); fill(-Inf,nc); zeros(nc)]
    ucon = [fill(Inf,nc);  ones(nc);     zeros(nc); zeros(nc)]
    K = ((1:4:4nc) .+ (0:3)')[:]
    lcon, ucon = lcon[K], ucon[K]
    lls  = LLSModel(A, b, C=C, lcon=lcon, ucon=ucon)
    lls2 = LLSModel(LinearOperator(A), b, C=C, lcon=lcon, ucon=ucon)
    lls3 = LLSModel(A, b, C=LinearOperator(C), lcon=lcon, ucon=ucon)
    lls4 = LLSModel(LinearOperator(A), b, C=LinearOperator(C), lcon=lcon, ucon=ucon)
    nlss = [lls, lls2, lls3, lls4]
    consistent_nls_counters(nlss)
    consistent_counters(nlss)
    consistent_nls_functions(nlss, exclude=[jac_residual, hess_coord_residual])
    consistent_nls_counters(nlss)
    consistent_counters(nlss)

    for nls in nlss
      reset!(nls)
    end
    nlss = [SlackNLSModel(nls) for nls in nlss]
    consistent_nls_counters(nlss)
    consistent_counters(nlss)
    consistent_nls_functions(nlss, exclude=[jac_residual, hess_coord_residual])
    consistent_nls_counters(nlss)
    consistent_counters(nlss)
  end

end
