import LinearAlgebra: I

export consistent_nlss

"""
    consistent_nlss(nlps; exclude=[hess, hprod, hess_coord])

Check that the all `nls`s of the vector `nlss` are consistent, in the sense that
- Their counters are the same.
- Their `meta` information is the same.
- The API functions return the same output given the same input.

In other words, if you create two models of the same problem, they should be consistent.

By default, the functions `hess`, `hprod` and `hess_coord` (and therefore associated functions) are excluded from this check, since some models don't implement them.
"""
function consistent_nlss(nlss; exclude=[hess, hess_coord, ghjvprod], test_slack=true, test_ff=true)
  consistent_nls_counters(nlss)
  consistent_counters(nlss)
  consistent_nls_functions(nlss, exclude=exclude)
  consistent_nls_counters(nlss)
  consistent_counters(nlss)
  for nls in nlss
    reset!(nls)
  end
  consistent_functions(nlss, exclude=exclude)

  if test_slack && has_inequalities(nlss[1])
    reset!.(nlss)
    slack_nlss = SlackNLSModel.(nlss)
    consistent_nls_functions(slack_nlss, exclude=exclude)
    consistent_nls_counters(slack_nlss)
    consistent_counters(slack_nlss)
    consistent_functions(slack_nlss, exclude=exclude)
  end

  if test_ff
    reset!.(nlss)
    ff_nlss = FeasibilityFormNLS.(nlss)
    consistent_nls_functions(ff_nlss, exclude=exclude)
    consistent_nls_counters(ff_nlss)
    consistent_counters(ff_nlss)
    consistent_functions(ff_nlss, exclude=exclude)
  end
end

function consistent_nls_counters(nlss)
  N = length(nlss)
  V = zeros(Int, N)
  for field in fieldnames(NLSCounters)
    field == :counters && continue
    @testset "Field $field" begin
      V = [eval(field)(nls) for nls in nlss]
      @test all(V .== V[1])
    end
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
      V = jac_coord_residual(nlss[i], x)
      I, J = jac_structure_residual(nlss[i])
      @test length(I) == length(J) == length(V) == nlss[i].nls_meta.nnzj
      I2, J2 = copy(I), copy(J)
      jac_structure_residual!(nlss[i], I2, J2)
      @test I == I2
      @test J == J2
      tmp_V = zeros(nlss[i].nls_meta.nnzj)
      jac_coord_residual!(nlss[i], x, tmp_V)
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

      rows, cols = jac_structure_residual(nlss[i])
      vals = jac_coord_residual(nlss[i], x)
      jprod_residual!(nlss[i], rows, cols, vals, v, tmp_m)
      @test isapprox(Jps[i], tmp_m, rtol=rtol)
      jprod_residual!(nlss[i], x, rows, cols, v, tmp_m)
      @test isapprox(Jps[i], tmp_m, rtol=rtol)

      J = jac_op_residual!(nlss[i], x, rows, cols, tmp_m, tmp_n)
      res = J * v
      @test isapprox(Jps[i], res, rtol=rtol)
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

      rows, cols = jac_structure_residual(nlss[i])
      vals = jac_coord_residual(nlss[i], x)
      jtprod_residual!(nlss[i], rows, cols, vals, v, tmp_n)
      @test isapprox(Jtps[i], tmp_n, rtol=rtol)
      jtprod_residual!(nlss[i], x, rows, cols, v, tmp_n)
      @test isapprox(Jtps[i], tmp_n, rtol=rtol)

      J = jac_op_residual!(nlss[i], x, rows, cols, tmp_m, tmp_n)
      res = J' * v
      @test isapprox(Jtps[i], res, rtol=rtol)
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
        V = hess_coord_residual(nlss[i], x, w)
        I, J = hess_structure_residual(nlss[i])
        @test length(I) == length(J) == length(V) == nlss[i].nls_meta.nnzh
        @test sparse(I, J, V, n, n) == Hs[i]
        I2, J2 = copy(I), copy(J)
        hess_structure_residual!(nlss[i], I2, J2)
        @test I == I2
        @test J == J2
        tmp_V = zeros(nlss[i].nls_meta.nnzh)
        hess_coord_residual!(nlss[i], x, w, tmp_V)
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
