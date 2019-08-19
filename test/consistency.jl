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
    V = [eval(field)(nlp) for nlp in nlps]
    @test all(V .== V[1])
  end
  V = [sum_counters(nlp) for nlp in nlps]
  @test all(V .== V[1])
end

function consistent_functions(nlps; rtol=1.0e-8, exclude=[])

  N = length(nlps)
  n = nlps[1].meta.nvar
  m = nlps[1].meta.ncon

  tmp_n = zeros(n)
  tmp_m = zeros(m)
  tmp_nn = zeros(n,n)

  x = 10 * [-(-1.0)^i for i = 1:n]

  if !(obj in exclude)
    fs = [obj(nlp, x) for nlp in nlps]
    fmin = minimum(map(abs, fs))
    for i = 1:N
      for j = i+1:N
        @test isapprox(fs[i], fs[j], atol=rtol * max(fmin, 1.0))
      end

      if !(objcons in exclude)
        # Test objcons for unconstrained problems
        if m == 0
          f, c = objcons(nlps[i], x)
          @test isapprox(fs[i], f, rtol=rtol)
          @test c == []
          f, tmpc = objcons!(nlps[i], x, c)
          @test isapprox(fs[i], f, rtol=rtol)
          @test c == []
          @test tmpc == []
        end
      end
    end
  end

  if !(grad in exclude)
    gs = Any[grad(nlp, x) for nlp in nlps]
    gmin = minimum(map(norm, gs))
    for i = 1:N
      for j = i+1:N
        @test isapprox(gs[i], gs[j], atol=rtol * max(gmin, 1.0))
      end
      tmpg = grad!(nlps[i], x, tmp_n)
      @test isapprox(gs[i], tmp_n, atol=rtol * max(gmin, 1.0))
      @test isapprox(tmpg, tmp_n, atol=rtol * max(gmin, 1.0))

      if !(objgrad in exclude)
        f, g = objgrad(nlps[i], x)
        @test isapprox(fs[i], f, atol=rtol * max(abs(f), 1.0))
        @test isapprox(gs[i], g, atol=rtol * max(gmin, 1.0))
        f, tmpg = objgrad!(nlps[i], x, g)
        @test isapprox(fs[i], f, atol=rtol * max(abs(f), 1.0))
        @test isapprox(gs[i], g, atol=rtol * max(gmin, 1.0))
        @test isapprox(g, tmpg, atol=rtol * max(gmin, 1.0))
      end
    end
  end

  if !(hess_coord in exclude)
    Hs = Vector{Any}(undef, N)
    for i = 1:N
      (I, J, V) = hess_coord(nlps[i], x)
      IS, JS = hess_structure(nlps[i])
      @test IS == I
      @test JS == J
      Hs[i] = sparse(I, J, V, n, n)
    end
    Hmin = minimum(map(norm, Hs))
    for i = 1:N
      for j = i+1:N
        @test isapprox(Hs[i], Hs[j], atol=rtol * max(Hmin, 1.0))
      end
      σ = 3.14
      (I, J, V) = hess_coord(nlps[i], x, obj_weight=σ)
      tmp_h = sparse(I, J, V, n, n)
      @test isapprox(σ*Hs[i], tmp_h, atol=rtol * max(Hmin, 1.0))
      tmp_V = zeros(nlps[i].meta.nnzh)
      hess_coord!(nlps[i], x, I, J, tmp_V, obj_weight=σ)
      @test tmp_V == V
    end
  end

  if !(hess in exclude)
    Hs = Any[hess(nlp, x) for nlp in nlps]
    Hmin = minimum(map(norm, Hs))
    for i = 1:N
      for j = i+1:N
        @test isapprox(Hs[i], Hs[j], atol=rtol * max(Hmin, 1.0))
      end
      σ = 3.14
      tmp_nn = hess(nlps[i], x, obj_weight=σ)
      @test isapprox(σ*Hs[i], tmp_nn, atol=rtol * max(Hmin, 1.0))
    end
  end

  v = 10 * [-(-1.0)^i for i = 1:n]

  if !(hprod in exclude)
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
      @test isapprox(tmphv, tmp_n, atol=rtol * max(Hvmin, 1.0))
      fill!(tmp_n, 0)
      H = hess_op!(nlps[i], x, tmp_n)
      res = H * v
      @test isapprox(res, Hvs[i], atol=rtol * max(Hvmin, 1.0))
      @test isapprox(res, tmp_n, atol=rtol * max(Hvmin, 1.0))
    end
  end

  if intersect([hess, hess_coord], exclude) == []
    for i = 1:N
      nlp = nlps[i]
      Hx = hess(nlp, x, obj_weight=0.5)
      I, J, V = hess_coord(nlp, x, obj_weight=0.5)
      @test length(I) == length(J) == length(V) == nlp.meta.nnzh
      @test sparse(I, J, V, n, n) == Hx
    end
  end

  if m > 0
    if !(cons in exclude)
      cs = Any[cons(nlp, x) for nlp in nlps]
      cls = [nlp.meta.lcon for nlp in nlps]
      cus = [nlp.meta.ucon for nlp in nlps]
      cmin = minimum(map(norm, cs))
      for i = 1:N
        tmpc = cons!(nlps[i], x, tmp_m)
        @test isapprox(cs[i], tmp_m, atol=rtol * max(cmin, 1.0))
        @test isapprox(tmpc, tmp_m, atol=rtol * max(cmin, 1.0))
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

        if !(objcons in exclude)
          f, c = objcons(nlps[i], x)
          @test isapprox(fs[i], f, atol=rtol * max(abs(f), 1.0))
          @test isapprox(cs[i],c, atol=rtol * max(cmin, 1.0))
          f, tmpc = objcons!(nlps[i], x, c)
          @test isapprox(fs[i], f, atol=rtol * max(abs(f), 1.0))
          @test isapprox(cs[i],c, atol=rtol * max(cmin, 1.0))
          @test isapprox(c, tmpc, atol=rtol * max(cmin, 1.0))
        end
      end
    end

    if intersect([jac, jac_coord], exclude) == []
      Js = [jac(nlp, x) for nlp in nlps]
      Jmin = minimum(map(norm, Js))
      for i = 1:N
        vi = norm(Js[i])
        for j = i+1:N
            @test isapprox(vi, norm(Js[j]), atol=rtol * max(Jmin, 1.0))
        end
        I, J, V = jac_coord(nlps[i], x)
        @test length(I) == length(J) == length(V) == nlps[i].meta.nnzj
        @test isapprox(sparse(I, J, V, m, n), Js[i], atol=rtol * max(Jmin, 1.0))
        IS, JS = jac_structure(nlps[i])
        @test IS == I
        @test JS == J
        tmp_V = zeros(nlps[i].meta.nnzj)
        jac_coord!(nlps[i], x, I, J, tmp_V)
        @test tmp_V == V
      end
    end

    if !(jprod in exclude)
      Jops = Any[jac_op(nlp, x) for nlp in nlps]
      Jps = Any[jprod(nlp, x, v) for nlp in nlps]
      for i = 1:N
        @test isapprox(Jps[i], Jops[i] * v, atol=rtol * max(Jmin, 1.0))
        vi = norm(Jps[i])
        for j = i+1:N
          @test isapprox(vi, norm(Jps[j]), atol=rtol * max(Jmin, 1.0))
        end
        tmpjv = jprod!(nlps[i], x, v, tmp_m)
        @test isapprox(tmpjv, tmp_m, atol=rtol * max(Jmin, 1.0))
        @test isapprox(Jps[i], tmp_m, atol=rtol * max(Jmin, 1.0))
        fill!(tmp_m, 0)
        J = jac_op!(nlps[i], x, tmp_m, tmp_n)
        res = J * v
        @test isapprox(res, Jps[i], atol=rtol * max(Jmin, 1.0))
        @test isapprox(res, tmp_m, atol=rtol * max(Jmin, 1.0))
      end
    end

    if !(jtprod in exclude)
      w = 10 * [-(-1.0)^i for i = 1:m]
      Jtps = Any[jtprod(nlp, x, w) for nlp in nlps]
      for i = 1:N
        @test isapprox(Jtps[i], Jops[i]' * w, atol=rtol * max(Jmin, 1.0))
        vi = norm(Jtps[i])
        for j = i+1:N
          @test isapprox(vi, norm(Jtps[j]), atol=rtol * max(Jmin, 1.0))
        end
        tmpjtv = jtprod!(nlps[i], x, w, tmp_n)
        @test isapprox(Jtps[i], tmp_n, atol=rtol * max(Jmin, 1.0))
        @test isapprox(tmpjtv, tmp_n, atol=rtol * max(Jmin, 1.0))
        fill!(tmp_n, 0)
        J = jac_op!(nlps[i], x, tmp_m, tmp_n)
        res = J' * w
        @test isapprox(res, Jtps[i], atol=rtol * max(Jmin, 1.0))
        @test isapprox(res, tmp_n, atol=rtol * max(Jmin, 1.0))
      end
    end

    y = 3.14 * ones(m)

    if !(hess_coord in exclude)
      Ls = Vector{Any}(undef, N)
      for i = 1:N
        (I, J, V) = hess_coord(nlps[i], x, y=y)
        IS, JS = hess_structure(nlps[i])
        @test IS == I
        @test JS == J
        Ls[i] = sparse(I, J, V, n, n)
      end
      Lmin = minimum(map(norm, Ls))
      for i = 1:N
        for j = i+1:N
          @test isapprox(Ls[i], Ls[j], atol=rtol * max(Lmin, 1.0))
        end
        σ = 3.14
        (I, J, V) = hess_coord(nlps[i], x, obj_weight=σ, y=σ*y)
        tmp_h = sparse(I, J, V, n, n)
        @test isapprox(σ*Ls[i], tmp_h, atol=rtol * max(Lmin, 1.0))
        tmp_V = zeros(nlps[i].meta.nnzh)
        hess_coord!(nlps[i], x, I, J, tmp_V, obj_weight=σ, y=σ*y)
        @test tmp_V == V
      end
    end

    if !(hess in exclude)
      Ls = Any[hess(nlp, x, y=y) for nlp in nlps]
      Lmin = minimum(map(norm, Ls))
      for i = 1:N
        for j = i+1:N
          @test isapprox(Ls[i], Ls[j], atol=rtol * max(Lmin, 1.0))
        end
        σ = 3.14
        tmp_nn = hess(nlps[i], x, obj_weight = σ, y=σ*y)
        @test isapprox(σ*Ls[i], tmp_nn, atol=rtol * max(Hmin, 1.0))
      end
    end

    if intersect([hess, hess_coord], exclude) == [] for i = 1:N
        nlp = nlps[i]
        Hx = hess(nlp, x, obj_weight=0.5, y=y)
        I, J, V = hess_coord(nlp, x, obj_weight=0.5, y=y)
        @test length(I) == length(J) == length(V) == nlp.meta.nnzh
        @test sparse(I, J, V, n, n) == Hx
      end
    end

    if !(hprod in exclude)
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

function consistent_nlps(nlps; rtol=1.0e-8)
  consistent_counters(nlps)
  consistent_meta(nlps, rtol=rtol)
  consistent_functions(nlps, rtol=rtol)
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

  # Test Quasi-Newton models
  qnmodels = [[LBFGSModel(nlp) for nlp in nlps];
              [LSR1Model(nlp) for nlp in nlps]]
  consistent_functions([nlps; qnmodels], exclude=[hess, hess_coord, hprod])
  consistent_counters([nlps; qnmodels])
  @printf("✓%12s", " ")

  # If there are inequalities, test the SlackModels of each of these models
  if nlps[1].meta.ncon > length(nlps[1].meta.jfix)
    slack_nlps = [SlackModel(nlp) for nlp in nlps]
    consistent_functions(slack_nlps)
    consistent_counters(slack_nlps)
    @printf("✓")
  else
    @printf("-")
  end
  @printf("\n")
end
