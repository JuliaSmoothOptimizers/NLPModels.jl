# an unconstrained problem should be returned unchanged
@printf("Checking slack formulation of genrose\t")
include("genrose.jl")
model = JuMPNLPModel(genrose())
smodel = SlackModel(model)
@assert smodel == model
@printf("✓\n")

# a bound-constrained problem should be returned unchanged
@printf("Checking slack formulation of hs5\t")
include("hs5.jl")
model = JuMPNLPModel(hs5())
smodel = SlackModel(model)
@assert smodel == model
@printf("✓\n")

# an equality-constrained problem should be returned unchanged
@printf("Checking slack formulation of hs6\t")
include("hs6.jl")
model = JuMPNLPModel(hs6())
smodel = SlackModel(model)
@assert smodel == model
@printf("✓\n")

# test problems that actually have inequality constraints

function check_slack_model(smodel)

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

  x = rand(N)
  s = x[n+1:N]
  y = rand(smodel.meta.ncon)

  # slack variables do not influence objective value
  @assert obj(model, x[1:n]) == obj(smodel, x)
  @assert neval_obj(model) == 2

  g = grad(model, x[1:n])
  G = grad(smodel, x)
  @assert all(g == G[1:n])
  @assert all(i -> (i == 0), G[n+1:N])
  @assert neval_grad(model) == 2

  h = hess(model, x[1:n], y=y)
  H = hess(smodel, x, y=y)
  @assert all(H[1:n, 1:n] == h)
  @assert all(i -> (i == 0), H[1:n, n+1:N])
  @assert all(i -> (i == 0), H[n+1:N, 1:n])
  @assert all(i -> (i == 0), H[n+1:N, n+1:N])
  @assert neval_hess(model) == 2

  v = rand(N)
  hv = hprod(model, x[1:n], v[1:n], y=y)
  HV = hprod(smodel, x, v, y=y)
  @assert all(HV[1:n] == hv)
  @assert all(i -> (i == 0), HV[n+1:N])
  @assert neval_hprod(model) == 2

  c = cons(model, x[1:n])
  C = cons(smodel, x)

  # slack variables do not influence equality constraints
  @assert all(C[jfix] == c[jfix])
  @assert all(C[jlow] == c[jlow] - s[1:nlow])
  @assert all(C[jupp] == c[jupp] - s[nlow+1:nlow+nupp])
  @assert all(C[jrng] == c[jrng] - s[nlow+nupp+1:nlow+nupp+nrng])
  @assert neval_cons(model) == 2

  j = jac(model, x[1:n])
  J = jac(smodel, x)
  K = J[:, n+1:N]
  @assert all(J[:, 1:n] == j)
  k = 1
  for l in collect([jlow ; jupp ; jrng])
    @assert J[l, n+k] == -1
    K[l, k] = 0
    k += 1
  end
  @assert all(i -> (i == 0), K)
  @assert neval_jac(model) == 2

  v = rand(N)
  Jv = J * v
  @assert all(jprod(smodel, x, v) == Jv)
  jv = zeros(smodel.meta.ncon)
  @assert all(jprod!(smodel, x, v, jv) == Jv)

  u = rand(smodel.meta.ncon)
  Jtu = J' * u
  @assert all(jtprod(smodel, x, u) == Jtu)
  jtu = zeros(N)
  @assert all(jtprod!(smodel, x, u, jtu) == Jtu)

  reset!(smodel)
end

for problem in [:hs10, :hs11, :hs14, :hs15]
  problem_s = string(problem)
  @printf("Checking slack formulation of %-8s\t", problem_s)
  include("$problem_s.jl")
  problem_f = eval(problem)
  nlp_jump = JuMPNLPModel(problem_f())
  slack_model = SlackModel(nlp_jump)
  check_slack_model(slack_model)
  @printf("✓\n")
end
