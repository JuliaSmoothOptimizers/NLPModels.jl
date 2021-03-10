using Test, NLPModels, NLPModelsCore, NLPModelsTestUtils, LinearAlgebra, LinearOperators, Printf, SparseArrays

problems = NLPModelsTestUtils.nlp_problems
nls_problems = NLPModelsTestUtils.nls_problems

# Including problems so that they won't be multiply loaded
for problem in problems ∪ [:GENROSE] # GENROSE does not have a manual version, so it's separate
  include("problems/$problem.jl")
end
for problem in nls_problems
  include("nls_problems/$problem.jl")
end

# ADNLPModel with no functions
model = ADNLPModel(x->dot(x,x), zeros(2), name="square")
@assert model.meta.name == "square"

include("test_slack_model.jl")
include("test_qn_model.jl")

include("nlp_testutils.jl")
include("nls_testutils.jl")

include("test_autodiff_model.jl")
include("test_nlsmodels.jl")
include("test_feasibility_form_nls.jl")