include(joinpath(Pkg.dir("NLPModels"), "test", "consistency.jl"))

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

function consistent_nls_functions(nlss; nloops=10, rtol=1.0e-8)
  N = length(nlss)
  n = nls_meta(nlss[1]).nvar
  m = nls_meta(nlss[1]).nequ

  tmp_n = zeros(n)
  tmp_m = zeros(m)

  for k = 1:nloops
    x = 10 * (rand(n) - 0.5)

    Fs = Any[residual(nls, x) for nls in nlss]
    for i = 1:N
      for j = i+1:N
        @test isapprox(Fs[i], Fs[j], rtol=rtol)
      end

      _ = residual!(nlss[i], x, tmp_m)
      @test _ == Fs[i]
      @test Fs[i] == tmp_m
    end

    Js = Any[jac_residual(nls, x) for nls in nlss]
    for i = 1:N
      for j = i+1:N
        @test isapprox(Js[i], Js[j], rtol=rtol)
      end
    end

    J_ops = Any[jac_op_residual(nls, x) for nls in nlss]
    Jv, Jtv = zeros(m), zeros(n)
    J_ops_inplace = Any[jac_op_residual!(nls, x, Jv, Jtv) for nls in nlss]

    v = rand(n)

    Jps = Any[jprod_residual(nls, x, v) for nls in nlss]
    for i = 1:N
      for j = i+1:N
        @test isapprox(Jps[i], Jps[j], rtol=rtol)
      end

      _ = jprod_residual!(nlss[i], x, v, tmp_m)
      @test _ == Jps[i]
      @test Jps[i] == tmp_m
      @test Jps[i] == J_ops[i] * v
      @test Jps[i] == J_ops_inplace[i] * v
    end

    v = rand(m)

    Jtps = Any[jtprod_residual(nls, x, v) for nls in nlss]
    for i = 1:N
      for j = i+1:N
        @test isapprox(Jtps[i], Jtps[j], rtol=rtol)
      end

      _ = jtprod_residual!(nlss[i], x, v, tmp_n)
      @test _ == Jtps[i]
      @test Jtps[i] == tmp_n
      @test Jtps[i] == J_ops[i]' * v
      @test Jtps[i] == J_ops_inplace[i]' * v
    end

    v = rand(n)

    for k = 1:m
      Hs = Any[hess_residual(nls, x, k) for nls in nlss]
      Hvs = Any[hprod_residual(nls, x, k, v) for nls in nlss]
      Hops = Any[hess_op_residual(nls, x, k) for nls in nlss]
      Hiv = zeros(n)
      Hops_inplace = Any[hess_op_residual!(nls, x, k, Hiv) for nls in nlss]
      for i = 1:N
        for j = i+1:N
          @test isapprox(Hs[i], Hs[j], rtol=rtol)
          @test isapprox(Hvs[i], Hvs[j], rtol=rtol)
        end

        _ = hprod_residual!(nlss[i], x, k, v, tmp_n)
        @test _ == Hvs[i]
        @test Hvs[i] == tmp_n
        @test Hvs[i] == Hops[i] * v
        @test Hvs[i] == Hops_inplace[i] * v
      end
    end

  end
end

function consistent_nls()
  @testset "Consistency of Linear problem" begin
    m, n = 50, 20
    A = rand(m, n)
    b = rand(m)
    lvar = -rand(n)
    uvar = rand(n)
    lls_model = LLSModel(A, b, lvar=lvar, uvar=uvar)
    simple_nls_model = SimpleNLSModel(zeros(n), m, lvar=lvar, uvar=uvar,
        F    = x->A*x-b,
        F!   = (x,Fx)->Fx[:]=A*x-b,
        J    = x->A,
        Jp   = (x,v)->A*v,
        Jp!  = (x,v,Jv)->Jv[:]=A*v,
        Jtp  = (x,v)->A'*v,
        Jtp! = (x,v,Jtv)->Jtv[:]=A'*v,
        Hi   = (x,i)->zeros(n,n),
        Hip  = (x,i,v)->zeros(n),
        Hip! = (x,i,v,Hiv)->fill!(Hiv, 0.0)
       )
    autodiff_model = ADNLSModel(x->A*x-b, zeros(n), m, lvar=lvar, uvar=uvar)
    nlp = ADNLPModel(x->0, zeros(n), lvar=lvar, uvar=uvar, c=x->A*x-b,
                     lcon=zeros(m), ucon=zeros(m))
    feas_res_model = FeasibilityResidual(nlp)
    nlss = [lls_model, simple_nls_model, autodiff_model, feas_res_model]
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
    consistent_functions(nlps, nloops=10)
  end

  @testset "Consistency of Nonlinear problem" begin
    m, n = 10, 2
    lvar = -rand(n)
    uvar = rand(n)
    F(x) = [2 + 2i - exp(i*x[1]) - exp(i*x[2]) for i = 1:m]
    F!(x,Fx) = (Fx[:] = F(x))
    x0 = [0.3; 0.4]
    J(x) = [-i*exp(i*x[j]) for i = 1:m, j = 1:2]
    Jp(x, v) = J(x)*v
    Jp!(x, v, Jv) = (Jv[:] = J(x)*v)
    Jtp(x, v) = J(x)'*v
    Jtp!(x, v, Jtv) = (Jtv[:] = J(x)'*v)
    Hi(x, i) = [-i^2*exp(i*x[1])  0.0; 0.0  -i^2*exp(i*x[2])]
    Hip(x, i, v) = -i^2*[exp(i*x[1])*v[1]; exp(i*x[2])*v[2]]
    Hip!(x, i, v, Hiv) = (Hiv[:] = -i^2*[exp(i*x[1])*v[1]; exp(i*x[2])*v[2]])

    simple_nls_model = SimpleNLSModel(x0, m, lvar=lvar, uvar=uvar, F=F, F! =F!, J=J, Jp=Jp, Jp! =Jp!, Jtp=Jtp, Jtp! =Jtp!, Hi=Hi, Hip=Hip, Hip! =Hip!)
    autodiff_model = ADNLSModel(F, x0, m, lvar=lvar, uvar=uvar)
    nlp = ADNLPModel(x->0, x0, lvar=lvar, uvar=uvar, c=F, lcon=zeros(m), ucon=zeros(m))
    feas_res_model = FeasibilityResidual(nlp)
    nlss = [simple_nls_model, autodiff_model, feas_res_model]
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
    consistent_functions(nlps, nloops=10)
  end
end

consistent_nls()
