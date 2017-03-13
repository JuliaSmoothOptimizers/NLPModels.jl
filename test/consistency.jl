function consistent_meta(nlps; rtol=1.0e-8)
  fields = [:nvar, :x0, :lvar, :uvar, :ifix, :ilow, :iupp, :irng, :ifree, :ncon,
    :y0]
  N = length(nlps)
  for field in fields
    for i = 1:N-1
      fi = getfield(nlps[i].meta, field)
      fj = getfield(nlps[i+1].meta, field)
      @test isapprox(fi, fj, rtol=rtol)
    end
  end
end

function consistent_counters(nlps)
  N = length(nlps)
  V = zeros(Int, N)
  for field in fieldnames(Counters)
    V = [getfield(nlp.counters, field) for nlp in nlps]
    @test all(V .== V[1])
  end
  V = [sum_counters(nlp) for nlp in nlps]
  @test all(V .== V[1])
end

function consistent_functions(nlps; nloops=100, rtol=1.0e-8)

  N = length(nlps)
  n = nlps[1].meta.nvar
  m = nlps[1].meta.ncon

  tmp_n = zeros(n)
  tmp_m = zeros(m)
  tmp_nn = zeros(n,n)

  for k = 1 : nloops
    x = 10 * (rand(n) - 0.5)

    fs = [obj(nlp, x) for nlp in nlps]
    fmin = minimum(map(abs, fs))
    for i = 1:N
      for j = i+1:N
        @test isapprox(fs[i], fs[j], atol=rtol * max(fmin, 1.0))
      end

      # Test objcons for unconstrained problems
      if m == 0
        f, c = objcons(nlps[i], x)
        @test fs[i] == f
        @test c == []
        f, tmpc = objcons!(nlps[i], x, c)
        @test fs[i] == f
        @test c == []
        @test tmpc == []
      end
    end

    gs = Any[grad(nlp, x) for nlp in nlps]
    gmin = minimum(map(norm, gs))
    for i = 1:N
      for j = i+1:N
        @test isapprox(gs[i], gs[j], atol=rtol * max(gmin, 1.0))
      end
      tmpg = grad!(nlps[i], x, tmp_n)
      @test isapprox(gs[i], tmp_n, atol=rtol * max(gmin, 1.0))
      @test tmpg == tmp_n

      f, g = objgrad(nlps[i], x)
      @test fs[i] == f
      @test gs[i] == g
      f, tmpg = objgrad!(nlps[i], x, g)
      @test fs[i] == f
      @test gs[i] == g
      @test g == tmpg
    end

    Hs = Array{Any}(N)
    for i = 1:N
      (I, J, V) = hess_coord(nlps[i], x)
      Hs[i] = sparse(I, J, V, n, n)
    end
    Hmin = minimum(map(vecnorm, Hs))
    for i = 1:N
      for j = i+1:N
        @test isapprox(Hs[i], Hs[j], atol=rtol * max(Hmin, 1.0))
      end
      σ = rand() - 0.5
      (I, J, V) = hess_coord(nlps[i], x, obj_weight=σ)
      tmp_h = sparse(I, J, V, n, n)
      @test isapprox(σ*Hs[i], tmp_h, atol=rtol * max(Hmin, 1.0))
    end

    Hs = Any[hess(nlp, x) for nlp in nlps]
    Hmin = minimum(map(vecnorm, Hs))
    for i = 1:N
      for j = i+1:N
        @test isapprox(Hs[i], Hs[j], atol=rtol * max(Hmin, 1.0))
      end
      σ = rand() - 0.5
      tmp_nn = hess(nlps[i], x, obj_weight=σ)
      @test isapprox(σ*Hs[i], tmp_nn, atol=rtol * max(Hmin, 1.0))
    end

    v = 10 * (rand(n) - 0.5)
    Hvs = Any[hprod(nlp, x, v) for nlp in nlps]
    Hopvs = Any[hess_op(nlp, x) * v for nlp in nlps]
    Hvmin = minimum(map(norm, Hvs))
    for i = 1:N
      for j = i+1:N
        @test isapprox(Hvs[i], Hvs[j], atol=rtol * max(Hvmin, 1.0))
        @test isapprox(Hvs[i], Hopvs[j], atol=rtol * max(Hvmin, 1.0))
      end
      tmphv = hprod!(nlps[i], x, v, tmp_n)
      @test isapprox(Hvs[i], tmp_n, atol=rtol * max(Hvmin, 1.0))
      @test tmphv == tmp_n
      fill!(tmp_n, 0)
      H = hess_op!(nlps[i], x, tmp_n)
      res = H * v
      @test isapprox(res, Hvs[i], atol=rtol * max(Hvmin, 1.0))
      @test isapprox(res, tmp_n, atol=rtol * max(Hvmin, 1.0))
    end

    if m > 0
      cs = Any[cons(nlp, x) for nlp in nlps]
      cls = [nlp.meta.lcon for nlp in nlps]
      cus = [nlp.meta.ucon for nlp in nlps]
      cmin = minimum(map(norm, cs))
      for i = 1:N
        tmpc = cons!(nlps[i], x, tmp_m)
        @test isapprox(cs[i], tmp_m, atol=rtol * max(cmin, 1.0))
        @test tmpc == tmp_m
        ci, li, ui = copy(cs[i]), cls[i], cus[i]
        for k = 1:m
          if li[k] > -Inf
            ci[k] -= li[k]
          elseif ui[k] < Inf
            ci[k] -= ui[k]
          end
        end
        for j = i+1:N
          cj, lj, uj = copy(cs[j]), cls[j], cus[j]
          for k = 1:m
            if lj[k] > -Inf
              cj[k] -= lj[k]
            elseif uj[k] < Inf
              cj[k] -= uj[k]
            end
          end
          @test isapprox(norm(ci), norm(cj), atol=rtol * max(cmin, 1.0))
        end

        f, c = objcons(nlps[i], x)
        @test fs[i] == f
        @test cs[i] == c
        f, tmpc = objcons!(nlps[i], x, c)
        @test fs[i] == f
        @test cs[i] == c
        @test c == tmpc
      end

      Js = Any[jac(nlp, x) for nlp in nlps]
      Jmin = minimum(map(vecnorm, Js))
      for i = 1:N-1
        vi = vecnorm(Js[i])
        for j = i+1:N
          @test isapprox(vi, vecnorm(Js[j]), atol=rtol * max(Jmin, 1.0))
        end
      end

      Jops = Any[jac_op(nlp, x) for nlp in nlps]
      Jps = Any[jprod(nlp, x, v) for nlp in nlps]
      for i = 1:N
        @test isapprox(Jps[i], Jops[i] * v, atol=rtol * max(Jmin, 1.0))
        vi = norm(Jps[i])
        for j = i+1:N
          @test isapprox(vi, norm(Jps[j]), atol=rtol * max(Jmin, 1.0))
        end
        tmpjv = jprod!(nlps[i], x, v, tmp_m)
        @test tmpjv == tmp_m
        @test isapprox(Jps[i], tmp_m, atol=rtol * max(Jmin, 1.0))
        fill!(tmp_m, 0)
        J = jac_op!(nlps[i], x, tmp_m, tmp_n)
        res = J * v
        @test isapprox(res, Jps[i], atol=rtol * max(Jmin, 1.0))
        @test isapprox(res, tmp_m, atol=rtol * max(Jmin, 1.0))
      end

      w = 10 * (rand() - 0.5) * ones(m)
      Jtps = Any[jtprod(nlp, x, w) for nlp in nlps]
      for i = 1:N
        @test isapprox(Jtps[i], Jops[i]' * w, atol=rtol * max(Jmin, 1.0))
        vi = norm(Jtps[i])
        for j = i+1:N
          @test isapprox(vi, norm(Jtps[j]), atol=rtol * max(Jmin, 1.0))
        end
        tmpjtv = jtprod!(nlps[i], x, w, tmp_n)
        @test isapprox(Jtps[i], tmp_n, atol=rtol * max(Jmin, 1.0))
        @test tmpjtv == tmp_n
        fill!(tmp_n, 0)
        J = jac_op!(nlps[i], x, tmp_m, tmp_n)
        res = J' * w
        @test isapprox(res, Jtps[i], atol=rtol * max(Jmin, 1.0))
        @test isapprox(res, tmp_n, atol=rtol * max(Jmin, 1.0))
      end

      y = (rand() - 0.5) * ones(m)

      Ls = Array{Any}(N)
      for i = 1:N
        (I, J, V) = hess_coord(nlps[i], x, y=y)
        Ls[i] = sparse(I, J, V, n, n)
      end
      Lmin = minimum(map(vecnorm, Ls))
      for i = 1:N
        for j = i+1:N
          @test isapprox(Ls[i], Ls[j], atol=rtol * max(Lmin, 1.0))
        end
        σ = rand() - 0.5
        (I, J, V) = hess_coord(nlps[i], x, obj_weight=σ, y=σ*y)
        tmp_h = sparse(I, J, V, n, n)
        @test isapprox(σ*Ls[i], tmp_h, atol=rtol * max(Lmin, 1.0))
      end

      Ls = Any[hess(nlp, x, y=y) for nlp in nlps]
      Lmin = minimum(map(vecnorm, Ls))
      for i = 1:N
        for j = i+1:N
          @test isapprox(Ls[i], Ls[j], atol=rtol * max(Lmin, 1.0))
        end
        σ = rand() - 0.5
        tmp_nn = hess(nlps[i], x, obj_weight=σ, y=σ*y)
        @test isapprox(σ*Ls[i], tmp_nn, atol=rtol * max(Hmin, 1.0))
      end

      Lps = Any[hprod(nlp, x, v, y=y) for nlp in nlps]
      Hopvs = Any[hess_op(nlp, x, y=y) * v for nlp in nlps]
      Lpmin = minimum(map(norm, Lps))
      for i = 1:N-1
        for j = i+1:N
          @test isapprox(Lps[i], Lps[j], atol=rtol * max(Lpmin, 1.0))
          @test isapprox(Lps[i], Hopvs[j], atol=rtol * max(Lpmin, 1.0))
        end
      end
    end
  end

end

function consistent_nlps(nlps; nloops=100, rtol=1.0e-8)
  consistent_counters(nlps)
  consistent_meta(nlps, rtol=rtol)
  consistent_functions(nlps, nloops=nloops, rtol=rtol)
  consistent_counters(nlps)
  for nlp in nlps
    reset!(nlp)
  end
  consistent_counters(nlps)
  @printf("✓%15s", " ")
  for nlp in nlps
    @assert length(gradient_check(nlp)) == 0
    @assert length(jacobian_check(nlp)) == 0
    @assert sum(map(length, values(hessian_check(nlp)))) == 0
    @assert sum(map(length, values(hessian_check_from_grad(nlp)))) == 0
  end
  @printf("✓%18s", " ")

  # If there are inequalities, test the SlackModels of each of these models
  if nlps[1].meta.ncon > length(nlps[1].meta.jfix)
    slack_nlps = [SlackModel(nlp) for nlp in nlps]
    consistent_functions(slack_nlps)
    @printf("✓")
  else
    @printf("-")
  end
  @printf("\n")
end

function consistency(problem :: Symbol; nloops=100, rtol=1.0e-8)
  problem_s = string(problem)
  @printf("Checking problem %-20s", problem_s)
  problem_f = eval(problem)
  nlp_autodiff = eval(parse("$(problem)_autodiff"))()
  nlp_mpb = MathProgNLPModel(problem_f())
  nlp_simple = eval(parse("$(problem)_simple"))()
  nlps = [nlp_autodiff; nlp_mpb; nlp_simple]

  consistent_nlps(nlps, nloops=nloops, rtol=rtol)
end
