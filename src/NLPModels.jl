module NLPModels

using FastClosures, LinearAlgebra, LinearOperators, NLPModelsCore, SparseArrays

include("autodiff_model.jl")
include("autodiff_nlsmodel.jl")
include("slack_model.jl")
include("qn_model.jl")
include("lls_model.jl")
include("feasibility_residual.jl")
include("feasibility_form_nls.jl")

end # module
