using LinearOperators

function check_qn_model(qnmodel)

  model = qnmodel.model
  @assert typeof(qnmodel) <: NLPModels.QuasiNewtonModel
  @assert qnmodel.meta.nvar == model.meta.nvar
  @assert qnmodel.meta.ncon == model.meta.ncon

  x = rand(qnmodel.meta.nvar)

  @assert obj(model, x) == obj(qnmodel, x)
  @assert neval_obj(model) == 2

  @assert grad(model, x) == grad(qnmodel, x)
  @assert neval_grad(model) == 2

  @assert cons(model, x) == cons(qnmodel, x)
  @assert neval_cons(model) == 2

  @assert jac(model, x) == jac(qnmodel, x)
  @assert neval_jac(model) == 2

  v = rand(qnmodel.meta.nvar)
  u = rand(qnmodel.meta.ncon)

  @assert jprod(model, x, v) == jprod(qnmodel, x, v)
  @assert neval_jprod(model) == 2

  @assert jtprod(model, x, u) == jtprod(qnmodel, x, u)
  @assert neval_jtprod(model) == 2

  H = hess_op(qnmodel, x)
  @assert typeof(H) <: LinearOperators.AbstractLinearOperator
  @assert size(H) == (model.meta.nvar, model.meta.nvar)
  @assert H * v == hprod(qnmodel, x, v)

  g = grad(qnmodel, x)
  gp = grad(qnmodel, x - g)
  push!(qnmodel, -g, gp - g)  # only testing that the call succeeds, not that the update is valid
  # the quasi-Newton operator itself is tested in LinearOperators

  reset!(qnmodel)
end

for problem in [:hs10, :hs11, :hs14, :hs15]
  problem_s = string(problem)
  include("$problem_s.jl")
  problem_f = eval(problem)
  nlp_jump = JuMPNLPModel(problem_f())
  @printf("Checking LBFGS formulation of %-8s\t", problem_s)
  qn_model = LBFGSModel(nlp_jump)
  check_qn_model(qn_model)
  @printf("✓\n")
  @printf("Checking LSR1 formulation of %-8s\t", problem_s)
  qn_model = LSR1Model(nlp_jump)
  check_qn_model(qn_model)
  @printf("✓\n")
end
