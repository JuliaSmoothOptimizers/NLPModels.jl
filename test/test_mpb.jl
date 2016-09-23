using Ipopt
using JuMP

include("hs6.jl")

# pass an AmplModel to IPOPT
nlp = MathProgNLPModel(hs6())
show(nlp.meta)
print(nlp.meta)
model = NLPtoMPB(nlp, IpoptSolver())
@assert isa(model, Ipopt.IpoptMathProgModel)
MathProgBase.optimize!(model)
@assert MathProgBase.getobjval(model) ≈ 0.0
