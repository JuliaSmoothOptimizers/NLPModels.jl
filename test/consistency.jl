using Base.Test
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

    Hs = Any[hess(nlp, x) for nlp in nlps]
    Hmin = minimum(map(vecnorm, Hs))
    for i = 1:N-1
      for j = i+1:N
        @test_approx_eq_eps Hs[i] Hs[j] rtol * max(Hmin, 1.0)
      end
      σ = rand() - 0.5
      tmp_nn = hess(nlps[i], x, obj_weight=σ)
      @test_approx_eq_eps σ*Hs[i] tmp_nn rtol * max(Hmin, 1.0)
    end

    v = 10 * (rand(n) - 0.5)
    Hvs = Any[hprod(nlp, x, v) for nlp in nlps]
    Hvmin = minimum(map(norm, Hvs))
    for i = 1:N
      for j = i+1:N
        @test_approx_eq_eps Hvs[i] Hvs[j] rtol * max(Hvmin, 1.0)
      end
      hprod!(nlps[i], x, v, tmp_n)
      @test_approx_eq_eps Hvs[i] tmp_n rtol * max(Hvmin, 1.0)
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

      Jps = Any[jprod(nlp, x, v) for nlp in nlps]
      Jpmin = minimum(map(norm, Jps))
      for i = 1:N
        vi = norm(Jps[i])
        for j = i+1:N
          @test_approx_eq_eps vi norm(Jps[j]) rtol * max(Jmin, 1.0)
        end
        jprod!(nlps[i], x, v, tmp_m)
        @test_approx_eq_eps Jps[i] tmp_m rtol * max(Jmin, 1.0)
      end

      w = 10 * (rand() - 0.5) * ones(m)
      Jtps = Any[jtprod(nlp, x, w) for nlp in nlps]
      Jtpmin = minimum(map(norm, Jps))
      for i = 1:N
        vi = norm(Jtps[i])
        for j = i+1:N
          @test_approx_eq_eps vi norm(Jtps[j]) rtol * max(Jmin, 1.0)
        end
        jtprod!(nlps[i], x, w, tmp_n)
        @test_approx_eq_eps Jtps[i] tmp_n rtol * max(Jmin, 1.0)
      end

      y = (rand() - 0.5) * ones(m)
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
      Lpmin = minimum(map(norm, Lps))
      for i = 1:N-1
        for j = i+1:N
          @test_approx_eq_eps Lps[i] Lps[j] rtol * max(Lpmin, 1.0)
        end
      end
    end
  end

end

function consistency(problem :: Symbol; nloops=100, rtol=1.0e-8)
  path = dirname(@__FILE__)
  problem_s = string(problem)
  @printf("Checking problem %-15s\t", problem_s)
  include("$problem_s.jl")
  problem_f = eval(problem)
  nlp_ampl = AmplModel(joinpath(path, "$problem_s.nl"))
  nlp_cutest = CUTEstModel(cutest_problem_name) # Variable defined in .jl
  nlp_jump = JuMPNLPModel(problem_f())
  nlp_simple = eval(parse("$(problem)_simple"))()
  nlps = [nlp_ampl; nlp_cutest; nlp_jump; nlp_simple]

  if nlp_ampl.meta.ncon == length(nlp_ampl.meta.jfix)
    for nlp in nlps[:]
      push!(nlps, SlackModel(nlp))
    end
  end

  consistent_meta(nlps, rtol=rtol)
  consistent_functions(nlps, nloops=nloops, rtol=rtol)
  @printf("✓\n")
  amplmodel_finalize(nlp_ampl)
  cutest_finalize(nlp_cutest)
end

problems = [:brownden, :hs005, :hs006, :hs010, :hs011, :hs014]
for problem in problems
  consistency(problem)
end
