function jump_vs_ampl_helper(nlp_jump, nlp_ampl; nloops=100, rtol=1.0e-10)

  n = nlp_ampl.meta.nvar
  m = nlp_ampl.meta.ncon

  for k = 1 : nloops
    x = 10 * (rand(n) - 0.5)

    f_jump = obj(nlp_jump, x)
    f_ampl = AmplNLReader.obj(nlp_ampl, x)
    @assert(abs(f_jump - f_ampl) <= rtol * max(abs(f_ampl), 1.0))

    g_jump = grad(nlp_jump, x)
    g_ampl = AmplNLReader.grad(nlp_ampl, x)
    @assert(norm(g_jump - g_ampl) <= rtol * max(norm(g_ampl), 1.0))

    # JuMP returns the lower triangle. AMPL returns the upper triangle.
    H_jump = hess(nlp_jump, x)
    H_ampl = AmplNLReader.hess(nlp_ampl, x)
    @assert(vecnorm(H_jump - H_ampl') <= rtol * max(vecnorm(H_ampl), 1.0))

    v = 10 * (rand(n) - 0.5)
    Hv_jump = hprod(nlp_jump, x, v)
    Hv_ampl = AmplNLReader.hprod(nlp_ampl, x, v)
    @assert(norm(Hv_jump - Hv_ampl) <= rtol * max(norm(Hv_ampl), 1.0))

    if m > 0
      c_jump = cons(nlp_jump, x)
      c_ampl = AmplNLReader.cons(nlp_ampl, x)
      @assert(norm(c_jump - c_ampl) <= rtol * max(norm(c_ampl), 1.0))

      J_jump = jac(nlp_jump, x)
      J_ampl = AmplNLReader.jac(nlp_ampl, x)
      @assert(vecnorm(J_jump - J_ampl) <= rtol * max(vecnorm(J_ampl), 1.0))

      y = 10 * (rand(m) - 0.5)

      # JuMP returns the lower triangle. AMPL returns the upper triangle.
      # MPB sets the Lagrangian to f + Σᵢ yᵢ cᵢ
      # AmplNLReader sets it to    f - Σᵢ yᵢ cᵢ
      H_jump = hess(nlp_jump, x, -y)
      H_ampl = AmplNLReader.hess(nlp_ampl, x, y=y)
      @assert(vecnorm(H_jump - H_ampl') <= rtol * max(vecnorm(H_ampl), 1.0))

      Hv_jump = hprod(nlp_jump, x, -y, v)
      Hv_ampl = AmplNLReader.hprod(nlp_ampl, x, v, y=y)
      @assert(norm(Hv_jump - Hv_ampl) <= rtol * max(norm(Hv_ampl), 1.0))
    end
  end
  
end

function jump_vs_ampl(problem :: Symbol; nloops=100, rtol=1.0e-10)

  problem_s = string(problem)
  @printf("Checking problem %-15s\t", problem_s)
  include("$problem_s.jl")
  problem_f = eval(problem)
  nlp_jump = NLPModel(problem_f())
  nlp_ampl = AmplNLReader.AmplModel("$problem_s.nl")
  jump_vs_ampl_helper(nlp_jump, nlp_ampl, nloops=nloops, rtol=rtol)
  @printf("[Ok]\n")
  AmplNLReader.amplmodel_finalize(nlp_ampl)
end

problems = [:genrose, :hs006]
for problem in problems
  jump_vs_ampl(problem)
end
