function jump_vs_simple_helper(nlp_jump, nlp_simple; nloops=100, rtol=1.0e-10)

  n = nlp_simple.meta.nvar
  m = nlp_simple.meta.ncon

  for k = 1 : nloops
    x = 10 * (rand(n) - 0.5)

    f_jump = obj(nlp_jump, x)
    f_simple = obj(nlp_simple, x)
    @assert(abs(f_jump - f_simple) <= rtol * max(abs(f_simple), 1.0))

    g_jump = grad(nlp_jump, x)
    g_simple = grad(nlp_simple, x)
    @assert(norm(g_jump - g_simple) <= rtol * max(norm(g_simple), 1.0))

    H_jump = hess(nlp_jump, x)
    H_simple = tril(hess(nlp_simple, x))
    @assert(vecnorm(H_jump - H_simple) <= rtol * max(vecnorm(H_simple), 1.0))

    v = 10 * (rand(n) - 0.5)
    Hv_jump = hprod(nlp_jump, x, v)
    Hv_simple = hprod(nlp_simple, x, v)
    @assert(norm(Hv_jump - Hv_simple) <= rtol * max(norm(Hv_simple), 1.0))

    if m > 0
      c_jump = cons(nlp_jump, x)
      c_simple = cons(nlp_simple, x)
      # JuMP subtracts the lhs or rhs of one-sided
      # nonlinear inequality and nonlinear equality constraints
      nln_low = ∩(nlp_simple.meta.nln, nlp_simple.meta.jlow)
      c_simple[nln_low] -= nlp_simple.meta.lcon[nln_low]
      nln_upp = ∩(nlp_simple.meta.nln, nlp_simple.meta.jupp)
      c_simple[nln_upp] -= nlp_simple.meta.ucon[nln_upp]
      nln_fix = ∩(nlp_simple.meta.nln, nlp_simple.meta.jfix)
      c_simple[nln_fix] -= nlp_simple.meta.lcon[nln_fix]
      @assert(norm(c_jump - c_simple) <= rtol * max(norm(c_simple), 1.0))

      J_jump = jac(nlp_jump, x)
      J_simple = jac(nlp_simple, x)
      @assert(vecnorm(J_jump - J_simple) <= rtol * max(vecnorm(J_simple), 1.0))

      y = 10 * (rand(m) - 0.5)

      H_jump = hess(nlp_jump, x, y=y)
      H_simple = hess(nlp_simple, x, y=y)
      @assert(vecnorm(H_jump - H_simple) <= rtol * max(vecnorm(H_simple), 1.0))

      Hv_jump = hprod(nlp_jump, x, v, y=y)
      Hv_simple = hprod(nlp_simple, x, v, y=y)
      @assert(norm(Hv_jump - Hv_simple) <= rtol * max(norm(Hv_simple), 1.0))
    end
  end

end

function jump_vs_simple(problem :: Symbol; nloops=100, rtol=1.0e-10)
  problem_s = string(problem)
  y0 = []
  @printf("Checking problem %-15s\t", problem_s)
  include("$problem_s.jl")
  problem_f = eval(problem)
  nlp_jump = JuMPNLPModel(problem_f())
  nlp_simple = eval(parse("$(problem)_simple"))()
  jump_vs_simple_helper(nlp_jump, nlp_simple, nloops=nloops, rtol=rtol)
  @printf("✓\n")
end

problems = [:genrose, :hs006]
for problem in problems
  jump_vs_simple(problem)
end
