export residual, residual!, jac_residual, jac_structure_residual, jac_structure_residual!
export jac_coord_residual!, jac_coord_residual, jprod_residual, jprod_residual!
export jtprod_residual, jtprod_residual!, jac_op_residual, jac_op_residual!
export hess_residual, hess_structure_residual, hess_structure_residual!
export hess_coord_residual!,
  hess_coord_residual, jth_hess_residual, jth_hess_residual_coord, jth_hess_residual_coord!
export hprod_residual, hprod_residual!, hess_op_residual, hess_op_residual!

"""
    Fx = residual(nls, x)

Computes ``F(x)``, the residual at x.
"""
function residual(nls::AbstractNLSModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nls.meta.nvar x
  Fx = S(undef, nls_meta(nls).nequ)
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
function jac_coord_residual(nls::AbstractNLSModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nls.meta.nvar x
  vals = S(undef, nls.nls_meta.nnzj)
  jac_coord_residual!(nls, x, vals)
end

"""
    Jv = jprod_residual(nls, x, v)

Computes the product of the Jacobian of the residual at x and a vector, i.e.,  ``J(x)v``.
"""
function jprod_residual(
  nls::AbstractNLSModel{T, S},
  x::AbstractVector,
  v::AbstractVector,
) where {T, S}
  @lencheck nls.meta.nvar x v
  Jv = S(undef, nls_meta(nls).nequ)
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
    Jtv = jtprod_residual(nls, x, v)

Computes the product of the transpose of the Jacobian of the residual at x and a vector, i.e.,  ``J(x)^Tv``.
"""
function jtprod_residual(
  nls::AbstractNLSModel{T, S},
  x::AbstractVector,
  v::AbstractVector,
) where {T, S}
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ v
  Jtv = S(undef, nls_meta(nls).nvar)
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
    Jx = jac_op_residual(nls, x)

Computes ``J(x)``, the Jacobian of the residual at x, in linear operator form.
"""
function jac_op_residual(nls::AbstractNLSModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nls.meta.nvar x
  Jv = S(undef, nls_meta(nls).nequ)
  Jtv = S(undef, nls.meta.nvar)
  return jac_op_residual!(nls, x, Jv, Jtv)
end

"""
    Jx = jac_op_residual!(nls, x, Jv, Jtv)

Computes ``J(x)``, the Jacobian of the residual at x, in linear operator form. The
vectors `Jv` and `Jtv` are used as preallocated storage for the operations.
"""
function jac_op_residual!(
  nls::AbstractNLSModel{T, S},
  x::AbstractVector,
  Jv::AbstractVector,
  Jtv::AbstractVector,
) where {T, S}
  @lencheck nls.meta.nvar x Jtv
  @lencheck nls.nls_meta.nequ Jv
  prod! = @closure (res, v, α, β) -> begin
    jprod_residual!(nls, x, v, Jv)
    if β == 0
      res .= α .* Jv
    else
      res .= α .* Jv .+ β .* res
    end
    return res
  end
  ctprod! = @closure (res, v, α, β) -> begin
    jtprod_residual!(nls, x, v, Jtv)
    if β == 0
      res .= α .* Jtv
    else
      res .= α .* Jtv .+ β .* res
    end
    return res
  end
  return LinearOperator{T}(
    nls_meta(nls).nequ,
    nls_meta(nls).nvar,
    false,
    false,
    prod!,
    ctprod!,
    ctprod!,
  )
end

"""
    Jx = jac_op_residual!(nls, rows, cols, vals, Jv, Jtv)

Computes ``J(x)``, the Jacobian of the residual given by `(rows, cols, vals)`, in linear operator form. The
vectors `Jv` and `Jtv` are used as preallocated storage for the operations.
"""
function jac_op_residual!(
  nls::AbstractNLSModel{T, S},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  Jv::AbstractVector,
  Jtv::AbstractVector,
) where {T, S}
  @lencheck nls.nls_meta.nnzj rows cols vals
  @lencheck nls.nls_meta.nequ Jv
  @lencheck nls.meta.nvar Jtv
  prod! = @closure (res, v, α, β) -> begin
    jprod_residual!(nls, rows, cols, vals, v, Jv)
    if β == 0
      res .= α .* Jv
    else
      res .= α .* Jv .+ β .* res
    end
    return res
  end
  ctprod! = @closure (res, v, α, β) -> begin
    jtprod_residual!(nls, rows, cols, vals, v, Jtv)
    if β == 0
      res .= α .* Jtv
    else
      res .= α .* Jtv .+ β .* res
    end
    return res
  end
  return LinearOperator{T}(
    nls_meta(nls).nequ,
    nls_meta(nls).nvar,
    false,
    false,
    prod!,
    ctprod!,
    ctprod!,
  )
end

"""
    H = hess_residual(nls, x, v)

Computes the linear combination of the Hessians of the residuals at `x` with coefficients
`v`.
A `Symmetric` object wrapping the lower triangle is returned.
"""
function hess_residual(nls::AbstractNLSModel, x::AbstractVector, v::AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ v
  rows, cols = hess_structure_residual(nls)
  vals = hess_coord_residual(nls, x, v)
  Symmetric(sparse(rows, cols, vals, nls.meta.nvar, nls.meta.nvar), :L)
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
function hess_coord_residual(
  nls::AbstractNLSModel{T, S},
  x::AbstractVector,
  v::AbstractVector,
) where {T, S}
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ v
  vals = S(undef, nls.nls_meta.nnzh)
  hess_coord_residual!(nls, x, v, vals)
end

"""
    Hj = jth_hess_residual(nls, x, j)

Computes the Hessian of the j-th residual at x.
"""
function jth_hess_residual(nls::AbstractNLSModel{T, S}, x::AbstractVector, j::Int) where {T, S}
  @lencheck nls.meta.nvar x
  @rangecheck 1 nls.nls_meta.nequ j
  rows, cols = hess_structure_residual(nls)
  vals = jth_hess_residual_coord(nls, x, j)
  return Symmetric(sparse(rows, cols, vals, nls.meta.nvar, nls.meta.nvar), :L)
end

"""
    vals = jth_hess_residual_coord(nls, x, j)

Evaluate the Hessian of j-th residual at `x` in sparse coordinate format.
Only the lower triangle is returned.
"""
function jth_hess_residual_coord(
  nls::AbstractNLSModel{T, S},
  x::AbstractVector,
  j::Int,
) where {T, S}
  @lencheck nls.meta.nvar x
  @rangecheck 1 nls.nls_meta.nequ j
  vals = S(undef, nls.nls_meta.nnzh)
  return jth_hess_residual_coord!(nls, x, j, vals)
end

"""
    vals = jth_hess_residual_coord!(nls, x, j, vals)

Evaluate the Hessian of j-th residual at `x` in sparse coordinate format, with `vals` of
length `nls.nls_meta.nnzh`, in place. Only the lower triangle is returned.
"""
function jth_hess_residual_coord!(
  nls::AbstractNLSModel{T, S},
  x::AbstractVector,
  j::Int,
  vals::AbstractVector,
) where {T, S}
  @lencheck nls.meta.nvar x
  @rangecheck 1 nls.nls_meta.nequ j
  @lencheck nls.nls_meta.nnzh vals
  increment!(nls, :neval_jhess_residual)
  decrement!(nls, :neval_hess_residual)
  v = [i == j ? one(T) : zero(T) for i = 1:(nls.nls_meta.nequ)]
  return hess_coord_residual!(nls, x, v, vals)
end

"""
    Hiv = hprod_residual(nls, x, i, v)

Computes the product of the Hessian of the i-th residual at x, times the vector v.
"""
function hprod_residual(
  nls::AbstractNLSModel{T, S},
  x::AbstractVector,
  i::Int,
  v::AbstractVector,
) where {T, S}
  @lencheck nls.meta.nvar x v
  @rangecheck 1 nls.nls_meta.nequ i
  Hv = S(undef, nls.meta.nvar)
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
function hess_op_residual(nls::AbstractNLSModel{T, S}, x::AbstractVector, i::Int) where {T, S}
  @lencheck nls.meta.nvar x
  @rangecheck 1 nls.nls_meta.nequ i
  Hiv = S(undef, nls.meta.nvar)
  return hess_op_residual!(nls, x, i, Hiv)
end

"""
    Hop = hess_op_residual!(nls, x, i, Hiv)

Computes the Hessian of the i-th residual at x, in linear operator form. The vector `Hiv` is used as preallocated storage for the operation.
"""
function hess_op_residual!(
  nls::AbstractNLSModel{T, S},
  x::AbstractVector,
  i::Int,
  Hiv::AbstractVector,
) where {T, S}
  @lencheck nls.meta.nvar x Hiv
  @rangecheck 1 nls.nls_meta.nequ i
  prod! = @closure (res, v, α, β) -> begin
    hprod_residual!(nls, x, i, v, Hiv)
    if β == 0
      res .= α .* Hiv
    else
      res .= α .* Hiv .+ β .* res
    end
    return res
  end
  return LinearOperator{T}(nls_meta(nls).nvar, nls_meta(nls).nvar, true, true, prod!, prod!, prod!)
end

"""
    f = obj(nls, x)
    f = obj(nls, x, Fx; recompute::Bool=true)

Evaluate `f(x)`, the objective function of `nls::AbstractNLSModel`. `Fx` is overwritten with the value of the residual `F(x)`.
If `recompute` is `true`, then `Fx` is updated with the residual at `x`.
"""
function obj(nls::AbstractNLSModel, x::AbstractVector, Fx::AbstractVector; recompute::Bool = true)
  @lencheck nls.meta.nvar x
  increment!(nls, :neval_obj)
  recompute && residual!(nls, x, Fx)
  return dot(Fx, Fx) / 2
end

function obj(nls::AbstractNLSModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nls.meta.nvar x
  Fx = S(undef, nls.nls_meta.nequ)
  return obj(nls, x, Fx)
end

"""
    f, c = objcons!(nls, x, c)
    f, c = objcons!(nls, x, c, Fx; recompute::Bool=true)

In-place evaluation of constraints and objective for AbstractNLSModel.
`Fx` is overwritten with the value of the residual `F(x)`.
If `recompute` is `true`, then `Fx` is updated with the residual at `x`.
"""
function objcons!(nls::AbstractNLSModel{T, S}, x::AbstractVector, c::AbstractVector) where {T, S}
  @lencheck nls.meta.nvar x
  @lencheck nls.meta.ncon c
  Fx = S(undef, nls.nls_meta.nequ)
  return objcons!(nls, x, c, Fx)
end

function objcons!(
  nls::AbstractNLSModel,
  x::AbstractVector,
  c::AbstractVector,
  Fx::AbstractVector;
  recompute::Bool = true,
)
  cons_nln!(nls, x, c)
  return obj(nls, x, Fx; recompute = recompute), c
end

"""
    g = grad!(nls, x, g)
    g = grad!(nls, x, g, Fx; recompute::Bool=true)

Evaluate `∇f(x)`, the gradient of the objective function of `nls::AbstractNLSModel` at `x` in place. `Fx` is overwritten with the value of the residual `F(x)`.
If `recompute` is `true`, then `Fx` is updated with the residual at `x`.
"""
function grad!(
  nls::AbstractNLSModel,
  x::AbstractVector,
  g::AbstractVector,
  Fx::AbstractVector;
  recompute::Bool = true,
)
  @lencheck nls.meta.nvar x g
  increment!(nls, :neval_grad)
  recompute && residual!(nls, x, Fx)
  return jtprod_residual!(nls, x, Fx, g)
end

function grad!(nls::AbstractNLSModel{T, S}, x::AbstractVector, g::AbstractVector) where {T, S}
  @lencheck nls.meta.nvar x g
  increment!(nls, :neval_grad)
  Fx = S(undef, nls.nls_meta.nequ)
  return grad!(nls, x, g, Fx)
end

"""
    f, g = objgrad!(nls, x, g)
    f, g = objgrad!(nls, x, g, Fx; recompute::Bool=true)

Evaluate f(x) and ∇f(x) of `nls::AbstractNLSModel` at `x`. `Fx` is overwritten with the value of the residual `F(x)`.
If `recompute` is `true`, then `Fx` is updated with the residual at `x`.
"""
function objgrad!(
  nls::AbstractNLSModel,
  x::AbstractVector,
  g::AbstractVector,
  Fx::AbstractVector;
  recompute::Bool = true,
)
  @lencheck nls.meta.nvar x g
  increment!(nls, :neval_obj)
  increment!(nls, :neval_grad)
  recompute && residual!(nls, x, Fx)
  jtprod_residual!(nls, x, Fx, g)
  return dot(Fx, Fx) / 2, g
end

function objgrad!(nls::AbstractNLSModel{T, S}, x::AbstractVector, g::AbstractVector) where {T, S}
  @lencheck nls.meta.nvar x g
  increment!(nls, :neval_obj)
  increment!(nls, :neval_grad)
  Fx = S(undef, nls.nls_meta.nequ)
  return objgrad!(nls, x, g, Fx)
end
