"""
Default implementations for AbstractNLSModel APIs.
"""

function residual(nls::AbstractNLSModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nls.meta.nvar x
  Fx = S(undef, nls_meta(nls).nequ)
  residual!(nls, x, Fx)
end

function jac_residual(nls::AbstractNLSModel, x::AbstractVector)
  @lencheck nls.meta.nvar x
  rows, cols = jac_structure_residual(nls)
  vals = jac_coord_residual(nls, x)
  sparse(rows, cols, vals, nls.nls_meta.nequ, nls.meta.nvar)
end

function jac_structure_residual(nls::AbstractNLSModel)
  rows = Vector{Int}(undef, nls.nls_meta.nnzj)
  cols = Vector{Int}(undef, nls.nls_meta.nnzj)
  jac_structure_residual!(nls, rows, cols)
end

function jac_coord_residual(nls::AbstractNLSModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nls.meta.nvar x
  vals = S(undef, nls.nls_meta.nnzj)
  jac_coord_residual!(nls, x, vals)
end

function jprod_residual(
  nls::AbstractNLSModel{T, S},
  x::AbstractVector,
  v::AbstractVector,
) where {T, S}
  @lencheck nls.meta.nvar x v
  Jv = S(undef, nls_meta(nls).nequ)
  jprod_residual!(nls, x, v, Jv)
end

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

function jac_op_residual(nls::AbstractNLSModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nls.meta.nvar x
  Jv = S(undef, nls_meta(nls).nequ)
  Jtv = S(undef, nls.meta.nvar)
  return jac_op_residual!(nls, x, Jv, Jtv)
end

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

function hess_residual(nls::AbstractNLSModel, x::AbstractVector, v::AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ v
  rows, cols = hess_structure_residual(nls)
  vals = hess_coord_residual(nls, x, v)
  Symmetric(sparse(rows, cols, vals, nls.meta.nvar, nls.meta.nvar), :L)
end

function hess_structure_residual(nls::AbstractNLSModel)
  rows = Vector{Int}(undef, nls.nls_meta.nnzh)
  cols = Vector{Int}(undef, nls.nls_meta.nnzh)
  hess_structure_residual!(nls, rows, cols)
end

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

function jth_hess_residual(nls::AbstractNLSModel{T, S}, x::AbstractVector, j::Int) where {T, S}
  @lencheck nls.meta.nvar x
  @rangecheck 1 nls.nls_meta.nequ j
  rows, cols = hess_structure_residual(nls)
  vals = jth_hess_residual_coord(nls, x, j)
  return Symmetric(sparse(rows, cols, vals, nls.meta.nvar, nls.meta.nvar), :L)
end

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

function hess_op_residual(nls::AbstractNLSModel{T, S}, x::AbstractVector, i::Int) where {T, S}
  @lencheck nls.meta.nvar x
  @rangecheck 1 nls.nls_meta.nequ i
  Hiv = S(undef, nls.meta.nvar)
  return hess_op_residual!(nls, x, i, Hiv)
end

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
  cons!(nls, x, c)
  return obj(nls, x, Fx; recompute = recompute), c
end

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
