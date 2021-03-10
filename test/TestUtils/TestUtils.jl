module TestUtils

using LinearAlgebra

using NLPModels

const nlp_problems = ["BROWNDEN", "HS5", "HS6", "HS10", "HS11", "HS14", "LINCON", "LINSV", "MGH01Feas"]
const nls_problems = ["LLS", "MGH01", "NLSHS20", "NLSLC"]

# Including problems so that they won't be multiply loaded
# GENROSE does not have a manual version, so it's separate
for problem in nlp_problems ∪ ["GENROSE"]
  include("nlp/problems/$(lowercase(problem)).jl")
end
for problem in nls_problems
  include("nls/problems/$(lowercase(problem)).jl")
end

end