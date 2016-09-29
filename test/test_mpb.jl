using Ipopt
using JuMP

# pass an ADNLPModel to IPOPT
nlp = ADNLPModel(x->dot(x,x), ones(2), c=x->[sum(x)-1], lcon=[0.0], ucon=[0.0])
show(nlp.meta)
print(nlp.meta)
model = NLPtoMPB(nlp, IpoptSolver())
@assert isa(model, Ipopt.IpoptMathProgModel)
MathProgBase.optimize!(model)
@assert MathProgBase.getobjval(model) ≈ 0.5

# Testing NLPtoMPB of a MathProgNLPModel
nlp = MathProgNLPModel(hs6())
NLPtoMPB(nlp, IpoptSolver())
