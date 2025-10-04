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
