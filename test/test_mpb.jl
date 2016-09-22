using Ipopt
using MathProgBase
using JuMP

# pass an AmplModel to IPOPT
nlp = JuMPNLPModel(hs6())
show(nlp.meta)
print(nlp.meta)
model = NLPtoMPB(nlp, IpoptSolver())
@assert isa(model, Ipopt.IpoptMathProgModel)
MathProgBase.optimize!(model)
@assert MathProgBase.getobjval(model) ≈ 0.0
