rtol = 1.0e-8

# Create a NLPModel from a JuMP model.
include("genrose.jl")
genrose_jump = NLPModel(genrose)

# Load corresponding AMPL model.
genrose_ampl = AmplNLReader.AmplModel("genrose.nl")

x0_ampl = genrose_ampl.meta.x0
x0_jump = genrose_jump.model.colVal
dx0 = norm(x0_ampl - x0_jump)
@printf("Initial points differ by %7.1e\n", dx0)
@assert(dx0 <= rtol * norm(x0_ampl))

f0_ampl = AmplNLReader.obj(genrose_ampl, x0_ampl)
f0_jump = obj(genrose_jump, x0_jump)
df0 = abs(f0_ampl - f0_jump)
@printf("Objective values at initial point differ by %7.1e\n", df0)
@assert(df0 <= rtol * abs(f0_ampl))

g0_ampl = AmplNLReader.grad(genrose_ampl, x0_ampl)
g0_jump = grad(genrose_jump, x0_jump)
dg0 = norm(g0_ampl - g0_jump)
@printf("Objective gradients at initial point differ by %7.1e\n", dg0)
@assert(dg0 <= rtol * norm(g0_ampl))

# JuMP returns the lower triangle. AMPL returns the upper triangle.
H0_ampl = AmplNLReader.hess(genrose_ampl, x0_ampl)
H0_jump = hess(genrose_jump, x0_jump)
dH0 = vecnorm(H0_ampl' - H0_jump)
@printf("Objective Hessians at initial point differ by %7.1e\n", dH0)
@assert(dH0 <= rtol * vecnorm(H0_ampl))

v = ones(genrose_ampl.meta.nvar)
hv0_ampl = AmplNLReader.hprod(genrose_ampl, x0_ampl, v)
hv0_jump = hprod(genrose_jump, x0_jump, v)
dhv0 = norm(hv0_ampl - hv0_jump)
@printf("Objective Hessian-vector products at initial point differ by %7.1e\n", dhv0)
@assert(dhv0 <= rtol * norm(hv0_ampl))
