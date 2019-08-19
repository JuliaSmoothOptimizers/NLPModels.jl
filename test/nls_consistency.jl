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

function consistent_nlss(nlss; rtol=1.0e-8)
  consistent_nls_counters(nlss)
  consistent_nls_functions(nlss, rtol=rtol)
  consistent_functions(nlss, rtol=rtol)
  consistent_nls_counters(nlss)
  for nls in nlss
    reset!(nls)
  end
  consistent_nls_counters(nlss)

  snlss = [SlackModel(nls) for nls in nlss]
  consistent_nls_counters(nlss)
  consistent_nls_functions(nlss, rtol=rtol)
  consistent_functions(nlss, rtol=rtol)
  consistent_nls_counters(nlss)
  for nls in nlss
    reset!(nls)
  end
  consistent_nls_counters(nlss)
end
