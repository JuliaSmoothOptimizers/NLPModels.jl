using LinearOperators

function check_qn_model(qnmodel)
  rtol  = 1e-8
  model = qnmodel.model
  @assert typeof(qnmodel) <: NLPModels.QuasiNewtonModel
  @assert qnmodel.meta.nvar == model.meta.nvar
  @assert qnmodel.meta.ncon == model.meta.ncon

  x = [-(-1.0)^i for i = 1:qnmodel.meta.nvar]

  @assert isapprox(obj(model, x), obj(qnmodel, x), rtol=rtol)
  @assert neval_obj(model) == 2

  @assert isapprox(grad(model, x), grad(qnmodel, x), rtol=rtol)
  @assert neval_grad(model) == 2

  @assert isapprox(cons(model, x), cons(qnmodel, x), rtol=rtol)
  @assert neval_cons(model) == 2

  @assert isapprox(jac(model, x), jac(qnmodel, x), rtol=rtol)
  @assert neval_jac(model) == 2

  v = [-(-1.0)^i for i = 1:qnmodel.meta.nvar]
  u = [-(-1.0)^i for i = 1:qnmodel.meta.ncon]

  @assert isapprox(jprod(model, x, v), jprod(qnmodel, x, v), rtol=rtol)
  @assert neval_jprod(model) == 2

  @assert isapprox(jtprod(model, x, u), jtprod(qnmodel, x, u), rtol=rtol)
  @assert neval_jtprod(model) == 2

  H = hess_op(qnmodel, x)
  @assert typeof(H) <: LinearOperators.AbstractLinearOperator
  @assert size(H) == (model.meta.nvar, model.meta.nvar)
  @assert isapprox(H * v, hprod(qnmodel, x, v), rtol=rtol)

  g = grad(qnmodel, x)
  gp = grad(qnmodel, x - g)
  push!(qnmodel, -g, gp - g)  # only testing that the call succeeds, not that the update is valid
  # the quasi-Newton operator itself is tested in LinearOperators

  reset!(qnmodel)
end

for problem in [:HS10, :HS11, :HS14]
  try
    eval(Symbol(problem))
  catch
    include("$problem.jl")
  end
  problem_f = eval(problem)
  nlp = problem_f()
  @printf("Checking LBFGS formulation of %-8s\t", problem)
  qn_model = LBFGSModel(nlp)
  check_qn_model(qn_model)
  @printf("✓\n")
  @printf("Checking LSR1 formulation of %-8s\t", problem)
  qn_model = LSR1Model(nlp)
  check_qn_model(qn_model)
  @printf("✓\n")
end
