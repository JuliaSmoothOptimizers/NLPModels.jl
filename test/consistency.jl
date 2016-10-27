function consistent_meta(nlps; rtol=1.0e-8)
  fields = [:nvar, :x0, :lvar, :uvar, :ifix, :ilow, :iupp, :irng, :ifree, :ncon,
    :y0]
  N = length(nlps)
  for field in fields
    for i = 1:N-1
      fi = getfield(nlps[i].meta, field)
      fj = getfield(nlps[i+1].meta, field)
      @test_approx_eq_eps fi fj rtol
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
    for i = 1:N-1
      for j = i+1:N
        @test_approx_eq_eps fs[i] fs[j] rtol * max(fmin, 1.0)
      end
    end

    gs = Any[grad(nlp, x) for nlp in nlps]
    gmin = minimum(map(norm, gs))
    for i = 1:N
      for j = i+1:N
        @test_approx_eq_eps gs[i] gs[j] rtol * max(gmin, 1.0)
      end
      grad!(nlps[i], x, tmp_n)
      @test_approx_eq_eps gs[i] tmp_n rtol * max(gmin, 1.0)
    end

    Hs = Array{Any}(N)
    for i = 1:N
      (I, J, V) = hess_coord(nlps[i], x)
      Hs[i] = sparse(I, J, V, n, n)
    end
    Hmin = minimum(map(vecnorm, Hs))
    for i = 1:N
      for j = i+1:N
        @test_approx_eq_eps Hs[i] Hs[j] rtol * max(Hmin, 1.0)
      end
      σ = rand() - 0.5
      (I, J, V) = hess_coord(nlps[i], x, obj_weight=σ)
      tmp_h = sparse(I, J, V, n, n)
      @test_approx_eq_eps σ*Hs[i] tmp_h rtol * max(Hmin, 1.0)
    end

    Hs = Any[hess(nlp, x) for nlp in nlps]
    Hmin = minimum(map(vecnorm, Hs))
    for i = 1:N
      for j = i+1:N
        @test_approx_eq_eps Hs[i] Hs[j] rtol * max(Hmin, 1.0)
      end
      σ = rand() - 0.5
      tmp_nn = hess(nlps[i], x, obj_weight=σ)
      @test_approx_eq_eps σ*Hs[i] tmp_nn rtol * max(Hmin, 1.0)
    end

    v = 10 * (rand(n) - 0.5)
    Hvs = Any[hprod(nlp, x, v) for nlp in nlps]
    Hopvs = Any[hess_op(nlp, x) * v for nlp in nlps]
    Hvmin = minimum(map(norm, Hvs))
    for i = 1:N
      for j = i+1:N
        @test_approx_eq_eps Hvs[i] Hvs[j] rtol * max(Hvmin, 1.0)
        @test_approx_eq_eps Hvs[i] Hopvs[j] rtol * max(Hvmin, 1.0)
      end
      hprod!(nlps[i], x, v, tmp_n)
      @test_approx_eq_eps Hvs[i] tmp_n rtol * max(Hvmin, 1.0)
      fill!(tmp_n, 0)
      H = hess_op!(nlps[i], x, tmp_n)
      res = H * v
      @test_approx_eq_eps res Hvs[i] rtol * max(Hvmin, 1.0)
      @test_approx_eq_eps res tmp_n rtol * max(Hvmin, 1.0)
    end

    if m > 0
      cs = Any[cons(nlp, x) for nlp in nlps]
      cls = [nlp.meta.lcon for nlp in nlps]
      cus = [nlp.meta.ucon for nlp in nlps]
      cmin = minimum(map(norm, cs))
      for i = 1:N
        cons!(nlps[i], x, tmp_m)
        @test_approx_eq_eps cs[i] tmp_m rtol * max(cmin, 1.0)
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
          @test_approx_eq_eps norm(ci) norm(cj) rtol * max(cmin, 1.0)
        end
      end

      Js = Any[jac(nlp, x) for nlp in nlps]
      Jmin = minimum(map(vecnorm, Js))
      for i = 1:N-1
        vi = vecnorm(Js[i])
        for j = i+1:N
          @test_approx_eq_eps vi vecnorm(Js[j]) rtol * max(Jmin, 1.0)
        end
      end

      Jops = Any[jac_op(nlp, x) for nlp in nlps]
      Jps = Any[jprod(nlp, x, v) for nlp in nlps]
      for i = 1:N
        @test_approx_eq_eps Jps[i] (Jops[i] * v) rtol * max(Jmin, 1.0)
        vi = norm(Jps[i])
        for j = i+1:N
          @test_approx_eq_eps vi norm(Jps[j]) rtol * max(Jmin, 1.0)
        end
        jprod!(nlps[i], x, v, tmp_m)
        @test_approx_eq_eps Jps[i] tmp_m rtol * max(Jmin, 1.0)
        fill!(tmp_m, 0)
        J = jac_op!(nlps[i], x, tmp_m, tmp_n)
        res = J * v
        @test_approx_eq_eps res Jps[i] rtol * max(Jmin, 1.0)
        @test_approx_eq_eps res tmp_m rtol * max(Jmin, 1.0)
      end

      w = 10 * (rand() - 0.5) * ones(m)
      Jtps = Any[jtprod(nlp, x, w) for nlp in nlps]
      for i = 1:N
        @test_approx_eq_eps Jtps[i] (Jops[i]' * w) rtol * max(Jmin, 1.0)
        vi = norm(Jtps[i])
        for j = i+1:N
          @test_approx_eq_eps vi norm(Jtps[j]) rtol * max(Jmin, 1.0)
        end
        jtprod!(nlps[i], x, w, tmp_n)
        @test_approx_eq_eps Jtps[i] tmp_n rtol * max(Jmin, 1.0)
        fill!(tmp_n, 0)
        J = jac_op!(nlps[i], x, tmp_m, tmp_n)
        res = J' * w
        @test_approx_eq_eps res Jtps[i] rtol * max(Jmin, 1.0)
        @test_approx_eq_eps res tmp_n rtol * max(Jmin, 1.0)
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
          @test_approx_eq_eps Ls[i] Ls[j] rtol * max(Lmin, 1.0)
        end
        σ = rand() - 0.5
        (I, J, V) = hess_coord(nlps[i], x, obj_weight=σ, y=σ*y)
        tmp_h = sparse(I, J, V, n, n)
        @test_approx_eq_eps σ*Ls[i] tmp_h rtol * max(Lmin, 1.0)
      end

      Ls = Any[hess(nlp, x, y=y) for nlp in nlps]
      Lmin = minimum(map(vecnorm, Ls))
      for i = 1:N
        for j = i+1:N
          @test_approx_eq_eps Ls[i] Ls[j] rtol * max(Lmin, 1.0)
        end
        σ = rand() - 0.5
        tmp_nn = hess(nlps[i], x, obj_weight=σ, y=σ*y)
        @test_approx_eq_eps σ*Ls[i] tmp_nn rtol * max(Hmin, 1.0)
      end

      Lps = Any[hprod(nlp, x, v, y=y) for nlp in nlps]
      Hopvs = Any[hess_op(nlp, x, y=y) * v for nlp in nlps]
      Lpmin = minimum(map(norm, Lps))
      for i = 1:N-1
        for j = i+1:N
          @test_approx_eq_eps Lps[i] Lps[j] rtol * max(Lpmin, 1.0)
          @test_approx_eq_eps Lps[i] Hopvs[j] rtol * max(Lpmin, 1.0)
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
