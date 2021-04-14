export residual, residual!, jac_residual, jac_structure_residual, jac_structure_residual!
export jac_coord_residual!, jac_coord_residual, jprod_residual, jprod_residual!
export jtprod_residual, jtprod_residual!, jac_op_residual, jac_op_residual!
export hess_residual, hess_structure_residual, hess_structure_residual!
export hess_coord_residual!, hess_coord_residual, jth_hess_residual
export hprod_residual, hprod_residual!, hess_op_residual, hess_op_residual!

"""
    Fx = residual(nls, x)

Computes ``F(x)``, the residual at x.
"""
function residual(nls::AbstractNLSModel, x::AbstractVector)
  @lencheck nls.meta.nvar x
  Fx = zeros(eltype(x), nls_meta(nls).nequ)
  residual!(nls, x, Fx)
end

"""
    Fx = residual!(nls, x, Fx)

Computes ``F(x)``, the residual at x.
"""
function residual! end

"""
    Jx = jac_residual(nls, x)

Computes ``J(x)``, the Jacobian of the residual at x.
"""
function jac_residual(nls::AbstractNLSModel, x::AbstractVector)
  @lencheck nls.meta.nvar x
  rows, cols = jac_structure_residual(nls)
  vals = jac_coord_residual(nls, x)
  sparse(rows, cols, vals, nls.nls_meta.nequ, nls.meta.nvar)
end

"""
    (rows,cols) = jac_structure_residual!(nls, rows, cols)

Returns the structure of the constraint's Jacobian in sparse coordinate format in place.
"""
function jac_structure_residual! end

"""
    (rows,cols) = jac_structure_residual(nls)

Returns the structure of the constraint's Jacobian in sparse coordinate format.
"""
function jac_structure_residual(nls::AbstractNLSModel)
  rows = Vector{Int}(undef, nls.nls_meta.nnzj)
  cols = Vector{Int}(undef, nls.nls_meta.nnzj)
  jac_structure_residual!(nls, rows, cols)
end

"""
    vals = jac_coord_residual!(nls, x, vals)

Computes the Jacobian of the residual at `x` in sparse coordinate format, rewriting
`vals`. `rows` and `cols` are not rewritten.
"""
function jac_coord_residual! end

"""
    (rows,cols,vals) = jac_coord_residual(nls, x)

Computes the Jacobian of the residual at `x` in sparse coordinate format.
"""
function jac_coord_residual(nls::AbstractNLSModel, x::AbstractVector)
  @lencheck nls.meta.nvar x
  vals = Vector{eltype(x)}(undef, nls.nls_meta.nnzj)
  jac_coord_residual!(nls, x, vals)
end

"""
    Jv = jprod_residual(nls, x, v)

Computes the product of the Jacobian of the residual at x and a vector, i.e.,  ``J(x)v``.
"""
function jprod_residual(nls::AbstractNLSModel, x::AbstractVector, v::AbstractVector)
  @lencheck nls.meta.nvar x v
  Jv = zeros(eltype(x), nls_meta(nls).nequ)
  jprod_residual!(nls, x, v, Jv)
end

"""
    Jv = jprod_residual!(nls, x, v, Jv)

Computes the product of the Jacobian of the residual at x and a vector, i.e.,  ``J(x)v``, storing it in `Jv`.
"""
function jprod_residual! end

"""
    Jv = jprod_residual!(nls, rows, cols, vals, v, Jv)

Computes the product of the Jacobian of the residual given by `(rows, cols, vals)`
and a vector, i.e.,  ``J(x)v``, storing it in `Jv`.
"""
function jprod_residual!(
  nls::AbstractNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck nls.nls_meta.nnzj rows cols vals
  @lencheck nls.meta.nvar v
  @lencheck nls.nls_meta.nequ Jv
  increment!(nls, :neval_jprod_residual)
  coo_prod!(rows, cols, vals, v, Jv)
end

"""
    Jv = jprod_residual!(nls, x, rows, cols, v, Jv)

Computes the product of the Jacobian of the residual at x and a vector, i.e.,  ``J(x)v``, storing it in `Jv`.
The structure of the Jacobian is given by `(rows, cols)`.
"""
function jprod_residual!(
  nls::AbstractNLSModel,
  x::AbstractVector,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck nls.meta.nvar x v
  @lencheck nls.nls_meta.nnzj rows cols
  @lencheck nls.nls_meta.nequ Jv
  jprod_residual!(nls, x, v, Jv)
end

"""
    Jtv = jtprod_residual(nls, x, v)

Computes the product of the transpose of the Jacobian of the residual at x and a vector, i.e.,  ``J(x)^Tv``.
"""
function jtprod_residual(nls::AbstractNLSModel, x::AbstractVector, v::AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ v
  Jtv = zeros(eltype(x), nls_meta(nls).nvar)
  jtprod_residual!(nls, x, v, Jtv)
end

"""
    Jtv = jtprod_residual!(nls, x, v, Jtv)

Computes the product of the transpose of the Jacobian of the residual at x and a vector, i.e.,  ``J(x)^Tv``, storing it in `Jtv`.
"""
function jtprod_residual! end

"""
    Jtv = jtprod_residual!(nls, rows, cols, vals, v, Jtv)

Computes the product of the transpose of the Jacobian of the residual given by `(rows, cols, vals)`
and a vector, i.e.,  ``J(x)^Tv``, storing it in `Jv`.
"""
function jtprod_residual!(
  nls::AbstractNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck nls.nls_meta.nnzj rows cols vals
  @lencheck nls.nls_meta.nequ v
  @lencheck nls.meta.nvar Jtv
  increment!(nls, :neval_jtprod_residual)
  coo_prod!(cols, rows, vals, v, Jtv)
end

"""
    Jtv = jtprod_residual!(nls, x, rows, cols, v, Jtv)

Computes the product of the transpose Jacobian of the residual at x and a vector, i.e.,  ``J(x)^Tv``, storing it in `Jv`.
The structure of the Jacobian is given by `(rows, cols)`.
"""
function jtprod_residual!(
  nls::AbstractNLSModel,
  x::AbstractVector,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck nls.meta.nvar x Jtv
  @lencheck nls.nls_meta.nnzj rows cols
  @lencheck nls.nls_meta.nequ v
  jtprod_residual!(nls, x, v, Jtv)
end

"""
    Jx = jac_op_residual(nls, x)

Computes ``J(x)``, the Jacobian of the residual at x, in linear operator form.
"""
function jac_op_residual(nls::AbstractNLSModel, x::AbstractVector)
  @lencheck nls.meta.nvar x
  prod = @closure v -> jprod_residual(nls, x, v)
  ctprod = @closure v -> jtprod_residual(nls, x, v)
  return LinearOperator{eltype(x)}(
    nls_meta(nls).nequ,
    nls_meta(nls).nvar,
    false,
    false,
    prod,
    ctprod,
    ctprod,
  )
end

"""
    Jx = jac_op_residual!(nls, x, Jv, Jtv)

Computes ``J(x)``, the Jacobian of the residual at x, in linear operator form. The
vectors `Jv` and `Jtv` are used as preallocated storage for the operations.
"""
function jac_op_residual!(
  nls::AbstractNLSModel,
  x::AbstractVector,
  Jv::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck nls.meta.nvar x Jtv
  @lencheck nls.nls_meta.nequ Jv
  prod = @closure v -> jprod_residual!(nls, x, v, Jv)
  ctprod = @closure v -> jtprod_residual!(nls, x, v, Jtv)
  return LinearOperator{eltype(x)}(
    nls_meta(nls).nequ,
    nls_meta(nls).nvar,
    false,
    false,
    prod,
    ctprod,
    ctprod,
  )
end

"""
    Jx = jac_op_residual!(nls, rows, cols, vals, Jv, Jtv)

Computes ``J(x)``, the Jacobian of the residual given by `(rows, cols, vals)`, in linear operator form. The
vectors `Jv` and `Jtv` are used as preallocated storage for the operations.
"""
function jac_op_residual!(
  nls::AbstractNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  Jv::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck nls.nls_meta.nnzj rows cols vals
  @lencheck nls.nls_meta.nequ Jv
  @lencheck nls.meta.nvar Jtv
  prod = @closure v -> jprod_residual!(nls, rows, cols, vals, v, Jv)
  ctprod = @closure v -> jtprod_residual!(nls, rows, cols, vals, v, Jtv)
  return LinearOperator{eltype(vals)}(
    nls_meta(nls).nequ,
    nls_meta(nls).nvar,
    false,
    false,
    prod,
    ctprod,
    ctprod,
  )
end

"""
    Jx = jac_op_residual!(nls, x, rows, cols, Jv, Jtv)

Computes ``J(x)``, the Jacobian of the residual at x, in linear operator form. The
vectors `Jv` and `Jtv` are used as preallocated storage for the operations.
The structure of the Jacobian should be given by `(rows, cols)`.
"""
function jac_op_residual!(
  nls::AbstractNLSModel,
  x::AbstractVector,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  Jv::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck nls.meta.nvar x Jtv
  @lencheck nls.nls_meta.nnzj rows cols
  @lencheck nls.nls_meta.nequ Jv
  vals = jac_coord_residual(nls, x)
  decrement!(nls, :neval_jac_residual)
  return jac_op_residual!(nls, rows, cols, vals, Jv, Jtv)
end

"""
    H = hess_residual(nls, x, v)

Computes the linear combination of the Hessians of the residuals at `x` with coefficients
`v`.
"""
function hess_residual(nls::AbstractNLSModel, x::AbstractVector, v::AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ v
  rows, cols = hess_structure_residual(nls)
  vals = hess_coord_residual(nls, x, v)
  sparse(rows, cols, vals, nls.meta.nvar, nls.meta.nvar)
end

"""
    (rows,cols) = hess_structure_residual(nls)

Returns the structure of the residual Hessian.
"""
function hess_structure_residual(nls::AbstractNLSModel)
  rows = Vector{Int}(undef, nls.nls_meta.nnzh)
  cols = Vector{Int}(undef, nls.nls_meta.nnzh)
  hess_structure_residual!(nls, rows, cols)
end

"""
    hess_structure_residual!(nls, rows, cols)

Returns the structure of the residual Hessian in place.
"""
function hess_structure_residual! end

"""
    vals = hess_coord_residual!(nls, x, v, vals)

Computes the linear combination of the Hessians of the residuals at `x` with coefficients
`v` in sparse coordinate format, rewriting `vals`.
"""
function hess_coord_residual! end

"""
    vals = hess_coord_residual(nls, x, v)

Computes the linear combination of the Hessians of the residuals at `x` with coefficients
`v` in sparse coordinate format.
"""
function hess_coord_residual(nls::AbstractNLSModel, x::AbstractVector, v::AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ v
  vals = Vector{eltype(x)}(undef, nls.nls_meta.nnzh)
  hess_coord_residual!(nls, x, v, vals)
end

"""
    Hj = jth_hess_residual(nls, x, j)

Computes the Hessian of the j-th residual at x.
"""
function jth_hess_residual(nls::AbstractNLSModel, x::AbstractVector, j::Int)
  @lencheck nls.meta.nvar x
  increment!(nls, :neval_jhess_residual)
  decrement!(nls, :neval_hess_residual)
  v = [i == j ? one(eltype(x)) : zero(eltype(x)) for i = 1:(nls.nls_meta.nequ)]
  return hess_residual(nls, x, v)
end

"""
    Hiv = hprod_residual(nls, x, i, v)

Computes the product of the Hessian of the i-th residual at x, times the vector v.
"""
function hprod_residual(nls::AbstractNLSModel, x::AbstractVector, i::Int, v::AbstractVector)
  @lencheck nls.meta.nvar x
  Hv = zeros(eltype(x), nls_meta(nls).nvar)
  hprod_residual!(nls, x, i, v, Hv)
end

"""
    Hiv = hprod_residual!(nls, x, i, v, Hiv)

Computes the product of the Hessian of the i-th residual at x, times the vector v, and stores it in vector Hiv.
"""
function hprod_residual! end

"""
    Hop = hess_op_residual(nls, x, i)

Computes the Hessian of the i-th residual at x, in linear operator form.
"""
function hess_op_residual(nls::AbstractNLSModel, x::AbstractVector, i::Int)
  @lencheck nls.meta.nvar x
  prod = @closure v -> hprod_residual(nls, x, i, v)
  return LinearOperator{eltype(x)}(
    nls_meta(nls).nvar,
    nls_meta(nls).nvar,
    true,
    true,
    prod,
    prod,
    prod,
  )
end

"""
    Hop = hess_op_residual!(nls, x, i, Hiv)

Computes the Hessian of the i-th residual at x, in linear operator form. The vector `Hiv` is used as preallocated storage for the operation.
"""
function hess_op_residual!(nls::AbstractNLSModel, x::AbstractVector, i::Int, Hiv::AbstractVector)
  @lencheck nls.meta.nvar x Hiv
  prod = @closure v -> hprod_residual!(nls, x, i, v, Hiv)
  return LinearOperator{eltype(x)}(
    nls_meta(nls).nvar,
    nls_meta(nls).nvar,
    true,
    true,
    prod,
    prod,
    prod,
  )
end

function obj(nls::AbstractNLSModel, x::AbstractVector)
  @lencheck nls.meta.nvar x
  increment!(nls, :neval_obj)
  Fx = residual(nls, x)
  return dot(Fx, Fx) / 2
end

function grad!(nls::AbstractNLSModel, x::AbstractVector, g::AbstractVector)
  @lencheck nls.meta.nvar x g
  increment!(nls, :neval_grad)
  Fx = residual(nls, x)
  return jtprod_residual!(nls, x, Fx, g)
end

function objgrad!(nls::AbstractNLSModel, x::AbstractVector, g::AbstractVector)
  @lencheck nls.meta.nvar x g
  increment!(nls, :neval_obj)
  increment!(nls, :neval_grad)
  Fx = residual(nls, x)
  jtprod_residual!(nls, x, Fx, g)
  return dot(Fx, Fx) / 2, g
end
