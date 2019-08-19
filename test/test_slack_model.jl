# an unconstrained problem should be returned unchanged
@printf("Checking slack formulation of genrose\t")
model = BROWNDEN()
smodel = SlackModel(model)
@assert smodel == model
@printf("✓\n")

# a bound-constrained problem should be returned unchanged
@printf("Checking slack formulation of hs5\t")
model = HS5()
smodel = SlackModel(model)
@assert smodel == model
@printf("✓\n")

# an equality-constrained problem should be returned unchanged
@printf("Checking slack formulation of hs6\t")
model = HS6()
smodel = SlackModel(model)
@assert smodel == model
@printf("✓\n")

# test problems that actually have inequality constraints

function check_slack_model(smodel)
  rtol  = 1e-8
  model = smodel.model
  @assert typeof(smodel) == NLPModels.SlackModel
  n = model.meta.nvar   # number of variables in original model
  N = smodel.meta.nvar  # number of variables in slack model
  jlow = model.meta.jlow; nlow = length(jlow)
  jupp = model.meta.jupp; nupp = length(jupp)
  jrng = model.meta.jrng; nrng = length(jrng)
  jfix = model.meta.jfix; nfix = length(jfix)

  @assert N == n + model.meta.ncon - nfix
  @assert smodel.meta.ncon == model.meta.ncon

  x = [-(-1.0)^i for i = 1:N]
  s = x[n+1:N]
  y = [-(-1.0)^i for i = 1:smodel.meta.ncon]

  # slack variables do not influence objective value
  @assert isapprox(obj(model, x[1:n]), obj(smodel, x), rtol=rtol)
  @assert neval_obj(model) == 2

  g = grad(model, x[1:n])
  G = grad(smodel, x)
  @assert isapprox(g, G[1:n], rtol=rtol)
  @assert all(i -> (i ≈ 0), G[n+1:N])
  @assert neval_grad(model) == 2

  h = hess(model, x[1:n], y=y)
  H = hess(smodel, x, y=y)
  @assert isapprox(H[1:n, 1:n], h, rtol=rtol)
  @assert all(i -> (i ≈ 0), H[1:n, n+1:N])
  @assert all(i -> (i ≈ 0), H[n+1:N, 1:n])
  @assert all(i -> (i ≈ 0), H[n+1:N, n+1:N])
  @assert neval_hess(model) == 2

  v = [-(-1.0)^i for i = 1:N]
  hv = hprod(model, x[1:n], v[1:n], y=y)
  HV = hprod(smodel, x, v, y=y)
  @assert isapprox(HV[1:n], hv, rtol=rtol)
  @assert all(i -> (i ≈ 0), HV[n+1:N])
  @assert neval_hprod(model) == 2

  c = cons(model, x[1:n])
  C = cons(smodel, x)

  # slack variables do not influence equality constraints
  @assert all(C[jfix] ≈ c[jfix])
  @assert all(C[jlow] ≈ c[jlow] - s[1:nlow])
  @assert all(C[jupp] ≈ c[jupp] - s[nlow+1:nlow+nupp])
  @assert all(C[jrng] ≈ c[jrng] - s[nlow+nupp+1:nlow+nupp+nrng])
  @assert neval_cons(model) == 2

  j = jac(model, x[1:n])
  J = jac(smodel, x)
  K = J[:, n+1:N]
  @assert all(J[:, 1:n] ≈ j)
  k = 1
  for l in collect([jlow ; jupp ; jrng])
    @assert J[l, n+k] ≈ -1
    K[l, k] = 0
    k += 1
  end
  @assert all(i -> (i ≈ 0), K)
  @assert neval_jac(model) == 2

  v = [-(-1.0)^i for i = 1:N]
  Jv = J * v
  @assert all(jprod(smodel, x, v) ≈ Jv)
  jv = zeros(smodel.meta.ncon)
  @assert all(jprod!(smodel, x, v, jv) ≈ Jv)

  u = [-(-1.0)^i for i = 1:smodel.meta.ncon]
  Jtu = J' * u
  @assert all(jtprod(smodel, x, u) ≈ Jtu)
  jtu = zeros(N)
  @assert all(jtprod!(smodel, x, u, jtu) ≈ Jtu)

  reset!(smodel)
end

for problem in [:HS10, :HS11, :HS14]
  @printf("Checking slack formulation of %-8s\t", problem)
  problem_f = eval(problem)
  nlp = problem_f()
  slack_model = SlackModel(nlp)
  check_slack_model(slack_model)
  @printf("✓\n")
end
