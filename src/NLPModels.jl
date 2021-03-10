module NLPModels

using LinearAlgebra, LinearOperators, Printf, SparseArrays, FastClosures

include("core/core.jl")

include("autodiff_model.jl")
include("autodiff_nlsmodel.jl")
include("feasibility_form_nls.jl")
include("feasibility_residual.jl")
include("lls_model.jl")
include("qn_model.jl")
include("slack_model.jl")

include("model-interaction.jl")

include("dercheck.jl")

end # module
