"""
	ManualDenseNLPModel <: AbstractDenseNLPModel

Concrete dense NLP model for demonstration and testing.
"""

export ManualDenseNLPModel

struct ManualDenseNLPModel{T, S} <: AbstractDenseNLPModel{T, S}
	jac::Matrix{T}
	hess::Matrix{T}
end

"""
	jac_coord(model::ManualDenseNLPModel, x)

Return the dense Jacobian (ignores x for demonstration).
"""
function jac_coord(model::ManualDenseNLPModel{T, S}, x::AbstractVector{T}) where {T, S}
	return model.jac
end

"""
	jac_structure(model::ManualDenseNLPModel)

Return the dense Jacobian structure (row/col indices).
"""
function jac_structure(model::ManualDenseNLPModel{T, S}) where {T, S}
	m, n = size(model.jac)
	rows = repeat(collect(1:m), inner=n)
	cols = repeat(collect(1:n), outer=m)
	return rows, cols
end

"""
	hess_coord(model::ManualDenseNLPModel, x)

Return the dense Hessian (ignores x for demonstration).
"""
function hess_coord(model::ManualDenseNLPModel{T, S}, x::AbstractVector{T}) where {T, S}
	return model.hess
end

"""
	hess_structure(model::ManualDenseNLPModel)

Return the dense Hessian structure (row/col indices).
"""
function hess_structure(model::ManualDenseNLPModel{T, S}) where {T, S}
	n, _ = size(model.hess)
	rows = repeat(collect(1:n), inner=n)
	cols = repeat(collect(1:n), outer=n)
	return rows, cols
end
