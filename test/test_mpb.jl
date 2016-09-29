using Ipopt
using JuMP

# pass an ADNLPModel to IPOPT
nlp = ADNLPModel(x->dot(x,x), ones(2), lvar=-ones(2), uvar=ones(2),
                 c=x->[sum(x)-1], lcon=[0.0], ucon=[0.0])
show(nlp.meta)
print(nlp.meta)
model = NLPtoMPB(nlp, IpoptSolver())
@assert isa(model, Ipopt.IpoptMathProgModel)
MathProgBase.optimize!(model)
@assert MathProgBase.getobjval(model) ≈ 0.5

# Calling functions to incrase coverage
d = NLPModelEvaluator(nlp)
Jv = zeros(1)
MathProgBase.eval_jac_prod(d, Jv, zeros(2), ones(2))
Jtv = zeros(2)
MathProgBase.eval_jac_prod_t(d, Jv, zeros(2), ones(1))
Hv = zeros(2)
MathProgBase.eval_hesslag_prod(d, Hv, zeros(2), ones(2), 1.0, ones(1))
MathProgBase.isobjlinear(d)
MathProgBase.isobjquadratic(d)
MathProgBase.isconstrlinear(d, 1)

# Testing NLPtoMPB of a MathProgNLPModel
nlp = MathProgNLPModel(hs6())
NLPtoMPB(nlp, IpoptSolver())
