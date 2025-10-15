# Default implementations for NLPModel API functions
function grad(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  g = S(undef, nlp.meta.nvar)
  return grad!(nlp, x, g)
end

function cons(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  c = S(undef, nlp.meta.ncon)
  return cons!(nlp, x, c)
end

function cons!(nlp::AbstractNLPModel, x::AbstractVector, cx::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon cx
  increment!(nlp, :neval_cons)
  if nlp.meta.nlin > 0
    if nlp.meta.nnln == 0
      cons_lin!(nlp, x, cx)
    else
      cons_lin!(nlp, x, view(cx, nlp.meta.lin))
    end
  end
  if nlp.meta.nnln > 0
    if nlp.meta.nlin == 0
      cons_nln!(nlp, x, cx)
    else
      cons_nln!(nlp, x, view(cx, nlp.meta.nln))
    end
  end
  return cx
end

function cons_lin(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  c = S(undef, nlp.meta.nlin)
  return cons_lin!(nlp, x, c)
end

function cons_nln(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  c = S(undef, nlp.meta.nnln)
  return cons_nln!(nlp, x, c)
end

function jth_congrad(nlp::AbstractNLPModel{T, S}, x::AbstractVector, j::Integer) where {T, S}
  @lencheck nlp.meta.nvar x
  g = S(undef, nlp.meta.nvar)
  return jth_congrad!(nlp, x, j, g)
end

function objcons(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  c = S(undef, nlp.meta.ncon)
  return objcons!(nlp, x, c)
end

function objcons!(nlp::AbstractNLPModel, x::AbstractVector, c::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon c
  f = obj(nlp, x)
  cons!(nlp, x, c)
  return f, c
end

function objgrad(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  g = S(undef, nlp.meta.nvar)
  return objgrad!(nlp, x, g)
end

function objgrad!(nlp::AbstractNLPModel, x::AbstractVector, g::AbstractVector)
  @lencheck nlp.meta.nvar x g
  f = obj(nlp, x)
  grad!(nlp, x, g)
  return f, g
end

function jac_structure(nlp::AbstractNLPModel)
  rows = Vector{Int}(undef, nlp.meta.nnzj)
  cols = Vector{Int}(undef, nlp.meta.nnzj)
  jac_structure!(nlp, rows, cols)
end

function jac_structure!(
  nlp::AbstractNLPModel,
  rows::AbstractVector{T},
  cols::AbstractVector{T},
) where {T}
  @lencheck nlp.meta.nnzj rows cols
  lin_ind = 1:(nlp.meta.lin_nnzj)
  if nlp.meta.nlin > 0
    if nlp.meta.nnln == 0
      jac_lin_structure!(nlp, rows, cols)
    else
      jac_lin_structure!(nlp, view(rows, lin_ind), view(cols, lin_ind))
      for i in lin_ind
        rows[i] += count(x < nlp.meta.lin[rows[i]] for x in nlp.meta.nln)
      end
    end
  end
  if nlp.meta.nnln > 0
    if nlp.meta.nlin == 0
      jac_nln_structure!(nlp, rows, cols)
    else
      nln_ind = (nlp.meta.lin_nnzj + 1):(nlp.meta.lin_nnzj + nlp.meta.nln_nnzj)
      jac_nln_structure!(nlp, view(rows, nln_ind), view(cols, nln_ind))
      for i in nln_ind
        rows[i] += count(x < nlp.meta.nln[rows[i]] for x in nlp.meta.lin)
      end
    end
  end
  return rows, cols
end

function jac_lin_structure(nlp::AbstractNLPModel)
  rows = Vector{Int}(undef, nlp.meta.lin_nnzj)
  cols = Vector{Int}(undef, nlp.meta.lin_nnzj)
  jac_lin_structure!(nlp, rows, cols)
end

function jac_nln_structure(nlp::AbstractNLPModel)
  rows = Vector{Int}(undef, nlp.meta.nln_nnzj)
  cols = Vector{Int}(undef, nlp.meta.nln_nnzj)
  jac_nln_structure!(nlp, rows, cols)
end

function jac_coord!(nlp::AbstractNLPModel, x::AbstractVector, vals::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzj vals
  increment!(nlp, :neval_jac)
  if nlp.meta.nlin > 0
    if nlp.meta.nnln == 0
      jac_lin_coord!(nlp, vals)
    else
      lin_ind = 1:(nlp.meta.lin_nnzj)
      jac_lin_coord!(nlp, view(vals, lin_ind))
    end
  end
  if nlp.meta.nnln > 0
    if nlp.meta.nlin == 0
      jac_nln_coord!(nlp, x, vals)
    else
      nln_ind = (nlp.meta.lin_nnzj + 1):(nlp.meta.lin_nnzj + nlp.meta.nln_nnzj)
      jac_nln_coord!(nlp, x, view(vals, nln_ind))
    end
  end
  return vals
end

function jac_coord(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  vals = S(undef, nlp.meta.nnzj)
  return jac_coord!(nlp, x, vals)
end

function jac(nlp::AbstractNLPModel, x::AbstractVector)
  @lencheck nlp.meta.nvar x
  rows, cols = jac_structure(nlp)
  vals = jac_coord(nlp, x)
  sparse(rows, cols, vals, nlp.meta.ncon, nlp.meta.nvar)
end

function jprod(nlp::AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x v
  Jv = S(undef, nlp.meta.ncon)
  return jprod!(nlp, x, v, Jv)
end

function jprod!(nlp::AbstractNLPModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck nlp.meta.nvar x v
  @lencheck nlp.meta.ncon Jv
  increment!(nlp, :neval_jprod)
  if nlp.meta.nlin > 0
    if nlp.meta.nnln == 0
      jprod_lin!(nlp, v, Jv)
    else
      jprod_lin!(nlp, v, view(Jv, nlp.meta.lin))
    end
  end
  if nlp.meta.nnln > 0
    if nlp.meta.nlin == 0
      jprod_nln!(nlp, x, v, Jv)
    else
      jprod_nln!(nlp, x, v, view(Jv, nlp.meta.nln))
    end
  end
  return Jv
end

function jprod!(
  nlp::AbstractNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck nlp.meta.nnzj rows cols vals
  @lencheck nlp.meta.nvar v
  @lencheck nlp.meta.ncon Jv
  increment!(nlp, :neval_jprod)
  coo_prod!(rows, cols, vals, v, Jv)
end

function jprod_lin(nlp::AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar v
  Jv = S(undef, nlp.meta.nlin)
  return jprod_lin!(nlp, v, Jv)
end

function jprod_lin!(
  nlp::AbstractNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck nlp.meta.lin_nnzj rows cols vals
  @lencheck nlp.meta.nvar v
  @lencheck nlp.meta.nlin Jv
  increment!(nlp, :neval_jprod_lin)
  coo_prod!(rows, cols, vals, v, Jv)
end

function jprod_nln(nlp::AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x v
  Jv = S(undef, nlp.meta.nnln)
  return jprod_nln!(nlp, x, v, Jv)
end

function jprod_nln!(
  nlp::AbstractNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck nlp.meta.nln_nnzj rows cols vals
  @lencheck nlp.meta.nvar v
  @lencheck nlp.meta.nnln Jv
  increment!(nlp, :neval_jprod_nln)
  coo_prod!(rows, cols, vals, v, Jv)
end


function hprod!(
  nlp::AbstractNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector,
)
  @lencheck nlp.meta.nnzh rows cols vals
  @lencheck nlp.meta.nvar v Hv
  increment!(nlp, :neval_hprod)
  coo_sym_prod!(cols, rows, vals, v, Hv)
end


function hess_op!(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x Hv
  prod! = @closure (res, v, α, β) -> begin
    hprod!(nlp, x, v, Hv; obj_weight = obj_weight)
    if β == 0
      res .= α .* Hv
    else
      res .= α .* Hv .+ β .* res
    end
    return res
  end
  return LinearOperator{T}(nlp.meta.nvar, nlp.meta.nvar, true, true, prod!, prod!, prod!)
end


function hess_op!(
  nlp::AbstractNLPModel{T, S},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  Hv::AbstractVector,
) where {T, S}
  @lencheck nlp.meta.nnzh rows cols vals
  @lencheck nlp.meta.nvar Hv
  prod! = @closure (res, v, α, β) -> begin
    hprod!(nlp, rows, cols, vals, v, Hv)
    if β == 0
      res .= α .* Hv
    else
      res .= α .* Hv .+ β .* res
    end
    return res
  end
  return LinearOperator{T}(nlp.meta.nvar, nlp.meta.nvar, true, true, prod!, prod!, prod!)
end


function hess_op!(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  y::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x Hv
  @lencheck nlp.meta.ncon y
  prod! = @closure (res, v, α, β) -> begin
    hprod!(nlp, x, y, v, Hv; obj_weight = obj_weight)
    if β == 0
      res .= α .* Hv
    else
      res .= α .* Hv .+ β .* res
    end
    return res
  end
  return LinearOperator{T}(nlp.meta.nvar, nlp.meta.nvar, true, true, prod!, prod!, prod!)
end


function jtprod(nlp::AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon v
  Jtv = S(undef, nlp.meta.nvar)
  return jtprod!(nlp, x, v, Jtv)
end

function jtprod!(nlp::AbstractNLPModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.ncon v
  increment!(nlp, :neval_jtprod)
  if nlp.meta.nnln == 0
    (nlp.meta.nlin > 0) && jtprod_lin!(nlp, v, Jtv)
  elseif nlp.meta.nlin == 0
    (nlp.meta.nnln > 0) && jtprod_nln!(nlp, x, v, Jtv)
  elseif nlp.meta.nlin >= nlp.meta.nnln
    jtprod_lin!(nlp, view(v, nlp.meta.lin), Jtv)
    if nlp.meta.nnln > 0
      Jtv .+= jtprod_nln(nlp, x, view(v, nlp.meta.nln))
    end
  else
    jtprod_nln!(nlp, x, view(v, nlp.meta.nln), Jtv)
    if nlp.meta.nlin > 0
      Jtv .+= jtprod_lin(nlp, view(v, nlp.meta.lin))
    end
  end
  return Jtv
end

function jtprod!(
  nlp::AbstractNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck nlp.meta.nnzj rows cols vals
  @lencheck nlp.meta.ncon v
  @lencheck nlp.meta.nvar Jtv
  increment!(nlp, :neval_jtprod)
  coo_prod!(cols, rows, vals, v, Jtv)
end

function jtprod_lin(nlp::AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
  @lencheck nlp.meta.nlin v
  Jtv = S(undef, nlp.meta.nvar)
  return jtprod_lin!(nlp, v, Jtv)
end

function jtprod_lin! end

function jtprod_lin!(
  nlp::AbstractNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck nlp.meta.lin_nnzj rows cols vals
  @lencheck nlp.meta.nlin v
  @lencheck nlp.meta.nvar Jtv
  increment!(nlp, :neval_jtprod_lin)
  coo_prod!(cols, rows, vals, v, Jtv)
end

function jtprod_nln(nlp::AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnln v
  Jtv = S(undef, nlp.meta.nvar)
  return jtprod_nln!(nlp, x, v, Jtv)
end

function jtprod_nln! end

function jtprod_nln!(
  nlp::AbstractNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck nlp.meta.nln_nnzj rows cols vals
  @lencheck nlp.meta.nnln v
  @lencheck nlp.meta.nvar Jtv
  increment!(nlp, :neval_jtprod_nln)
  coo_prod!(cols, rows, vals, v, Jtv)
end


function jac_op(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  Jv = S(undef, nlp.meta.ncon)
  Jtv = S(undef, nlp.meta.nvar)
  return jac_op!(nlp, x, Jv, Jtv)
end

function jac_op!(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector{T},
  Jv::AbstractVector,
  Jtv::AbstractVector,
) where {T, S}
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.ncon Jv
  prod! = @closure (res, v, α, β) -> begin # res = α * J * v + β * res
    jprod!(nlp, x, v, Jv)
    if β == 0
      res .= α .* Jv
    else
      res .= α .* Jv .+ β .* res
    end
    return res
  end
  ctprod! = @closure (res, v, α, β) -> begin
    jtprod!(nlp, x, v, Jtv)
    if β == 0
      res .= α .* Jtv
    else
      res .= α .* Jtv .+ β .* res
    end
    return res
  end
  return LinearOperator{T}(nlp.meta.ncon, nlp.meta.nvar, false, false, prod!, ctprod!, ctprod!)
end

function jac_op!(
  nlp::AbstractNLPModel{T, S},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector{T},
  Jv::AbstractVector,
  Jtv::AbstractVector,
) where {T, S}
  @lencheck nlp.meta.nnzj rows cols vals
  @lencheck nlp.meta.ncon Jv
  @lencheck nlp.meta.nvar Jtv
  prod! = @closure (res, v, α, β) -> begin # res = α * J * v + β * res
    jprod!(nlp, rows, cols, vals, v, Jv)
    if β == 0
      res .= α .* Jv
    else
      res .= α .* Jv .+ β .* res
    end
    return res
  end
  ctprod! = @closure (res, v, α, β) -> begin
    jtprod!(nlp, rows, cols, vals, v, Jtv)
    if β == 0
      res .= α .* Jtv
    else
      res .= α .* Jtv .+ β .* res
    end
    return res
  end
  return LinearOperator{T}(nlp.meta.ncon, nlp.meta.nvar, false, false, prod!, ctprod!, ctprod!)
end

function jac_lin_op(nlp::AbstractNLPModel{T, S}) where {T, S}
  Jv = S(undef, nlp.meta.nlin)
  Jtv = S(undef, nlp.meta.nvar)
  return jac_lin_op!(nlp, Jv, Jtv)
end

function jac_lin_op!(
  nlp::AbstractNLPModel{T, S},
  Jv::AbstractVector,
  Jtv::AbstractVector,
) where {T, S}
  @lencheck nlp.meta.nlin Jv
  @lencheck nlp.meta.nvar Jtv
  prod! = @closure (res, v, α, β) -> begin # res = α * J * v + β * res
    jprod_lin!(nlp, v, Jv)
    if β == 0
      res .= α .* Jv
    else
      res .= α .* Jv .+ β .* res
    end
    return res
  end
  ctprod! = @closure (res, v, α, β) -> begin
    jtprod_lin!(nlp, v, Jtv)
    if β == 0
      res .= α .* Jtv
    else
      res .= α .* Jtv .+ β .* res
    end
    return res
  end
  return LinearOperator{T}(nlp.meta.nlin, nlp.meta.nvar, false, false, prod!, ctprod!, ctprod!)
end

function jac_lin_op!(
  nlp::AbstractNLPModel{T, S},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector{T},
  Jv::AbstractVector,
  Jtv::AbstractVector,
) where {T, S}
  @lencheck nlp.meta.lin_nnzj rows cols vals
  @lencheck nlp.meta.nlin Jv
  @lencheck nlp.meta.nvar Jtv
  prod! = @closure (res, v, α, β) -> begin # res = α * J * v + β * res
    jprod_lin!(nlp, rows, cols, vals, v, Jv)
    if β == 0
      res .= α .* Jv
    else
      res .= α .* Jv .+ β .* res
    end
    return res
  end
  ctprod! = @closure (res, v, α, β) -> begin
    jtprod_lin!(nlp, rows, cols, vals, v, Jtv)
    if β == 0
      res .= α .* Jtv
    else
      res .= α .* Jtv .+ β .* res
    end
    return res
  end
  return LinearOperator{T}(nlp.meta.nlin, nlp.meta.nvar, false, false, prod!, ctprod!, ctprod!)
end

function jac_nln_op(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  Jv = S(undef, nlp.meta.nnln)
  Jtv = S(undef, nlp.meta.nvar)
  return jac_nln_op!(nlp, x, Jv, Jtv)
end

function jac_nln_op!(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector{T},
  Jv::AbstractVector,
  Jtv::AbstractVector,
) where {T, S}
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.nnln Jv
  prod! = @closure (res, v, α, β) -> begin # res = α * J * v + β * res
    jprod_nln!(nlp, x, v, Jv)
    if β == 0
      res .= α .* Jv
    else
      res .= α .* Jv .+ β .* res
    end
    return res
  end
  ctprod! = @closure (res, v, α, β) -> begin
    jtprod_nln!(nlp, x, v, Jtv)
    if β == 0
      res .= α .* Jtv
    else
      res .= α .* Jtv .+ β .* res
    end
    return res
  end
  return LinearOperator{T}(nlp.meta.nnln, nlp.meta.nvar, false, false, prod!, ctprod!, ctprod!)
end

function jac_nln_op!(
  nlp::AbstractNLPModel{T, S},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector{T},
  Jv::AbstractVector,
  Jtv::AbstractVector,
) where {T, S}
  @lencheck nlp.meta.nln_nnzj rows cols vals
  @lencheck nlp.meta.nnln Jv
  @lencheck nlp.meta.nvar Jtv
  prod! = @closure (res, v, α, β) -> begin # res = α * J * v + β * res
    jprod_nln!(nlp, rows, cols, vals, v, Jv)
    if β == 0
      res .= α .* Jv
    else
      res .= α .* Jv .+ β .* res
    end
    return res
  end
  ctprod! = @closure (res, v, α, β) -> begin
    jtprod_nln!(nlp, rows, cols, vals, v, Jtv)
    if β == 0
      res .= α .* Jtv
    else
      res .= α .* Jtv .+ β .* res
    end
    return res
  end
  return LinearOperator{T}(nlp.meta.nnln, nlp.meta.nvar, false, false, prod!, ctprod!, ctprod!)
end


