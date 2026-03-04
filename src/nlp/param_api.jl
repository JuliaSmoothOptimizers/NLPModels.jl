export grad_param, grad_param!,
    jac_param_structure, jac_param_structure!,
    jac_param_coord, jac_param_coord!,
    jpprod, jpprod!,
    jptprod, jptprod!,
    hess_param_structure, hess_param_structure!,
    hess_param_coord, hess_param_coord!,
    hpprod, hpprod!,
    hptprod, hptprod!,
    lcon_jac_param_structure, lcon_jac_param_structure!,
    lcon_jac_param_coord, lcon_jac_param_coord!,
    lcon_jpprod, lcon_jpprod!,
    lcon_jptprod, lcon_jptprod!,
    ucon_jac_param_structure, ucon_jac_param_structure!,
    ucon_jac_param_coord, ucon_jac_param_coord!,
    ucon_jpprod, ucon_jpprod!,
    ucon_jptprod, ucon_jptprod!,
    lvar_jac_param_structure, lvar_jac_param_structure!,
    lvar_jac_param_coord, lvar_jac_param_coord!,
    lvar_jpprod, lvar_jpprod!,
    lvar_jptprod, lvar_jptprod!,
    uvar_jac_param_structure, uvar_jac_param_structure!,
    uvar_jac_param_coord, uvar_jac_param_coord!,
    uvar_jpprod, uvar_jpprod!,
    uvar_jptprod, uvar_jptprod!


"""
    g = grad_param(nlp, x)

Evaluate `∇ₚf(x)`, the gradient of the objective function at `x` wrt parameters.
This function is only available if `nlp.meta.grad_param_available` is set to `true`.
"""
function grad_param(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck get_nvar(nlp.meta) x
  g = S(undef, get_nparam(nlp.meta))
  return grad_param!(nlp, x, g)
end

"""
    g = grad_param!(nlp, x, g)

Evaluate `∇ₚf(x)`, the gradient of the objective function at `x` wrt parameters in place.
This function is only available if `nlp.meta.grad_param_available` is set to `true`.
"""
function grad_param! end

"""
    (rows, cols) = jac_param_structure(nlp)

Return the structure of the constraints Jacobian wrt parameters in sparse coordinate format.
This function is only available if `nlp.meta.jac_param_available` is set to `true`.
"""
function jac_param_structure(nlp::AbstractNLPModel)
  rows = Vector{Int}(undef, get_nnzjp(nlp.meta))
  cols = Vector{Int}(undef, get_nnzjp(nlp.meta))
  jac_param_structure!(nlp, rows, cols)
end

"""
    jac_param_structure!(nlp, rows, cols)

Return the structure of the constraints Jacobian wrt parameters in sparse coordinate format in place.
This function is only available if `nlp.meta.jac_param_available` is set to `true`.
"""
function jac_param_structure! end

"""
    vals = jac_param_coord!(nlp, x, vals)

Evaluate ``Jₚ(x)``, the constraints Jacobian wrt parameters at `x` in sparse coordinate format, rewriting `vals`.
This function is only available if `nlp.meta.jac_param_available` is set to `true`.
"""
function jac_param_coord! end

"""
    vals = jac_param_coord(nlp, x)

Evaluate ``Jₚ(x)``, the constraints Jacobian wrt parameters at `x` in sparse coordinate format.
This function is only available if `nlp.meta.jac_param_available` is set to `true`.
"""
function jac_param_coord(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck get_nvar(nlp.meta) x
  vals = S(undef, get_nnzjp(nlp.meta))
  return jac_param_coord!(nlp, x, vals)
end

"""
    Jv = jpprod(nlp, x, v)

Evaluate ``Jₚ(x)v``, the parametric Jacobian-vector product at `x`.
This function is only available if `nlp.meta.jpprod_available` is set to `true`.
"""
function jpprod(nlp::AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector) where {T, S}
  @lencheck get_nvar(nlp.meta) x
  @lencheck get_nparam(nlp.meta) v
  Jv = S(undef, get_ncon(nlp.meta))
  return jpprod!(nlp, x, v, Jv)
end

"""
    Jv = jpprod!(nlp, x, v, Jv)

Evaluate ``Jₚ(x)v``, the parametric Jacobian-vector product at `x` in place.
This function is only available if `nlp.meta.jpprod_available` is set to `true`.
"""
function jpprod! end

"""
    Jtv = jptprod(nlp, x, v)

Evaluate ``Jₚ(x)ᵀv``, the parametric Jacobian-transpose vector product at `x`.
This function is only available if `nlp.meta.jptprod_available` is set to `true`.
"""
function jptprod(nlp::AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector) where {T, S}
  @lencheck get_nvar(nlp.meta) x
  @lencheck get_ncon(nlp.meta) v
  Jtv = S(undef, get_nparam(nlp.meta))
  return jptprod!(nlp, x, v, Jtv)
end

"""
    Jtv = jptprod!(nlp, x, v, Jtv)

Evaluate ``Jₚ(x)ᵀv``, the parametric Jacobian-transpose vector product at `x` in place.
This function is only available if `nlp.meta.jptprod_available` is set to `true`.
"""
function jptprod! end

"""
    (rows, cols) = hess_param_structure(nlp)

Return the structure of the Lagrangian variable-parameter Hessian in sparse coordinate format.
This function is only available if `nlp.meta.hess_param_available` is set to `true`.
"""
function hess_param_structure(nlp::AbstractNLPModel)
  rows = Vector{Int}(undef, get_nnzhp(nlp.meta))
  cols = Vector{Int}(undef, get_nnzhp(nlp.meta))
  hess_param_structure!(nlp, rows, cols)
end

"""
    hess_param_structure!(nlp, rows, cols)

Return the structure of the Lagrangian variable-parameter Hessian in sparse coordinate format in place.
This function is only available if `nlp.meta.hess_param_available` is set to `true`.
"""
function hess_param_structure! end

"""
    vals = hess_param_coord!(nlp, x, vals; obj_weight=1.0)

Evaluate the variable-parameter objective Hessian at `x` in sparse coordinate format,
with objective function scaled by `obj_weight`, overwriting `vals`.
This function is only available if `nlp.meta.hess_param_available` is set to `true`.
"""
function hess_param_coord!(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector{T},
  vals::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck get_nvar(nlp.meta) x
  @lencheck get_nnzhp(nlp.meta) vals
  y = fill!(S(undef, get_ncon(nlp.meta)), 0)
  hess_param_coord!(nlp, x, y, vals, obj_weight = obj_weight)
end

"""
    vals = hess_param_coord!(nlp, x, y, vals; obj_weight=1.0)

Evaluate the Lagrangian variable-parameter Hessian at `(x,y)` in sparse coordinate format,
with objective function scaled by `obj_weight`, overwriting `vals`.
This function is only available if `nlp.meta.hess_param_available` is set to `true`.
"""
function hess_param_coord! end

"""
    vals = hess_param_coord(nlp, x; obj_weight=1.0)

Evaluate the variable-parameter objective Hessian at `x` in sparse coordinate format,
with objective function scaled by `obj_weight`.
This function is only available if `nlp.meta.hess_param_available` is set to `true`.
"""
function hess_param_coord(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck get_nvar(nlp.meta) x
  vals = S(undef, get_nnzhp(nlp.meta))
  return hess_param_coord!(nlp, x, vals; obj_weight = obj_weight)
end

"""
    vals = hess_param_coord(nlp, x, y; obj_weight=1.0)

Evaluate the Lagrangian variable-parameter Hessian at `(x,y)` in sparse coordinate format,
with objective function scaled by `obj_weight`.
This function is only available if `nlp.meta.hess_param_available` is set to `true`.
"""
function hess_param_coord(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  y::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck get_nvar(nlp.meta) x
  @lencheck get_ncon(nlp.meta) y
  vals = S(undef, get_nnzhp(nlp.meta))
  return hess_param_coord!(nlp, x, y, vals; obj_weight = obj_weight)
end

"""
    Hv = hpprod(nlp, x, v; obj_weight=1.0)

Evaluate the product of the objective variable-parameter Hessian at `x` with the vector `v`,
with objective function scaled by `obj_weight`.
This function is only available if `nlp.meta.hpprod_available` is set to `true`.
"""
function hpprod(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  v::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck get_nvar(nlp.meta) x
  @lencheck get_nparam(nlp.meta) v
  Hv = S(undef, get_nvar(nlp.meta))
  return hpprod!(nlp, x, v, Hv; obj_weight = obj_weight)
end

"""
    Hv = hpprod(nlp, x, y, v; obj_weight=1.0)

Evaluate the product of the Lagrangian variable-parameter Hessian at `(x,y)` with the vector `v`,
with objective function scaled by `obj_weight`.
This function is only available if `nlp.meta.hpprod_available` is set to `true`.
"""
function hpprod(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck get_nvar(nlp.meta) x
  @lencheck get_nparam(nlp.meta) v
  @lencheck get_ncon(nlp.meta) y
  Hv = S(undef, get_nvar(nlp.meta))
  return hpprod!(nlp, x, y, v, Hv; obj_weight = obj_weight)
end

"""
    Hv = hpprod!(nlp, x, v, Hv; obj_weight=1.0)

Evaluate the product of the objective variable-parameter Hessian at `x` with the vector `v` in
place, with objective function scaled by `obj_weight`.
This function is only available if `nlp.meta.hpprod_available` is set to `true`.
"""
function hpprod!(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck get_nvar(nlp.meta) x Hv
  @lencheck get_nparam(nlp.meta) v
  y = fill!(S(undef, get_ncon(nlp.meta)), 0)
  hpprod!(nlp, x, y, v, Hv, obj_weight = obj_weight)
end

"""
    Hv = hpprod!(nlp, x, y, v, Hv; obj_weight=1.0)

Evaluate the product of the Lagrangian variable-parameter Hessian at `(x,y)` with the vector `v` in
place, with objective function scaled by `obj_weight`.
This function is only available if `nlp.meta.hpprod_available` is set to `true`.
"""
function hpprod! end

"""
    Htv = hptprod(nlp, x, y, v; obj_weight=1.0)

Evaluate the product of the Lagrangian variable-parameter Hessian transpose at `(x,y)` with `v`,
with objective function scaled by `obj_weight`.
This function is only available if `nlp.meta.hptprod_available` is set to `true`.
"""
function hptprod(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck get_nvar(nlp.meta) x v
  @lencheck get_ncon(nlp.meta) y
  Htv = S(undef, get_nparam(nlp.meta))
  return hptprod!(nlp, x, y, v, Htv; obj_weight = obj_weight)
end

"""
    Htv = hptprod!(nlp, x, y, v, Htv; obj_weight=1.0)

Evaluate the product of the Lagrangian variable-parameter Hessian transpose at `(x,y)` with `v` in
place, with objective function scaled by `obj_weight`.
This function is only available if `nlp.meta.hptprod_available` is set to `true`.
"""
function hptprod! end

"""
    (rows, cols) = lcon_jac_param_structure(nlp)

Return the structure of the constraint lower bound Jacobian wrt parameters in sparse coordinate format.
"""
function lcon_jac_param_structure(nlp::AbstractNLPModel)
  rows = Vector{Int}(undef, get_nnzjplcon(nlp.meta))
  cols = Vector{Int}(undef, get_nnzjplcon(nlp.meta))
  lcon_jac_param_structure!(nlp, rows, cols)
end

function lcon_jac_param_structure! end

"""
    vals = lcon_jac_param_coord!(nlp, vals)

Evaluate the lower constraint-bound Jacobian wrt parameters in sparse coordinate format, overwriting `vals`.
"""
function lcon_jac_param_coord! end

"""
    vals = lcon_jac_param_coord(nlp)

Evaluate the lower constraint-bound Jacobian wrt parameters in sparse coordinate format.
"""
function lcon_jac_param_coord(nlp::AbstractNLPModel{T, S}) where {T, S}
  vals = S(undef, get_nnzjplcon(nlp.meta))
  return lcon_jac_param_coord!(nlp, vals)
end

"""
    Jv = lcon_jpprod(nlp, v)

Evaluate `∇ₚlcon ⋅ v`, the lower constraint-bound Jacobian product.
"""
function lcon_jpprod(nlp::AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
  @lencheck get_nparam(nlp.meta) v
  Jv = S(undef, get_ncon(nlp.meta))
  return lcon_jpprod!(nlp, v, Jv)
end

function lcon_jpprod! end

"""
    Jtv = lcon_jptprod(nlp, v)

Evaluate `∇ₚlconᵀ ⋅ v`, the lower constraint-bound Jacobian transpose product.
"""
function lcon_jptprod(nlp::AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
  @lencheck get_ncon(nlp.meta) v
  Jtv = S(undef, get_nparam(nlp.meta))
  return lcon_jptprod!(nlp, v, Jtv)
end

function lcon_jptprod! end

"""
    (rows, cols) = ucon_jac_param_structure(nlp)

Return the structure of the constraint upper bound Jacobian wrt parameters in sparse coordinate format.
"""
function ucon_jac_param_structure(nlp::AbstractNLPModel)
  rows = Vector{Int}(undef, get_nnzjpucon(nlp.meta))
  cols = Vector{Int}(undef, get_nnzjpucon(nlp.meta))
  ucon_jac_param_structure!(nlp, rows, cols)
end

function ucon_jac_param_structure! end

"""
    vals = ucon_jac_param_coord!(nlp, vals)

Evaluate the upper constraint-bound Jacobian wrt parameters in sparse coordinate format, overwriting `vals`.
"""
function ucon_jac_param_coord! end

"""
    vals = ucon_jac_param_coord(nlp)

Evaluate the upper constraint-bound Jacobian wrt parameters in sparse coordinate format.
"""
function ucon_jac_param_coord(nlp::AbstractNLPModel{T, S}) where {T, S}
  vals = S(undef, get_nnzjpucon(nlp.meta))
  return ucon_jac_param_coord!(nlp, vals)
end

"""
    Jv = ucon_jpprod(nlp, v)

Evaluate `∇ₚucon ⋅ v`, the upper constraint-bound Jacobian product.
"""
function ucon_jpprod(nlp::AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
  @lencheck get_nparam(nlp.meta) v
  Jv = S(undef, get_ncon(nlp.meta))
  return ucon_jpprod!(nlp, v, Jv)
end

function ucon_jpprod! end

"""
    Jtv = ucon_jptprod(nlp, v)

Evaluate `∇ₚuconᵀ ⋅ v`, the upper constraint-bound Jacobian transpose product.
"""
function ucon_jptprod(nlp::AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
  @lencheck get_ncon(nlp.meta) v
  Jtv = S(undef, get_nparam(nlp.meta))
  return ucon_jptprod!(nlp, v, Jtv)
end

function ucon_jptprod! end

"""
    (rows, cols) = lvar_jac_param_structure(nlp)

Return the structure of the variable lower bound Jacobian wrt parameters in sparse coordinate format.
"""
function lvar_jac_param_structure(nlp::AbstractNLPModel)
  rows = Vector{Int}(undef, get_nnzjplvar(nlp.meta))
  cols = Vector{Int}(undef, get_nnzjplvar(nlp.meta))
  lvar_jac_param_structure!(nlp, rows, cols)
end

function lvar_jac_param_structure! end

"""
    vals = lvar_jac_param_coord!(nlp, vals)

Evaluate the lower variable-bound Jacobian wrt parameters in sparse coordinate format, overwriting `vals`.
"""
function lvar_jac_param_coord! end

"""
    vals = lvar_jac_param_coord(nlp)

Evaluate the lower variable-bound Jacobian wrt parameters in sparse coordinate format.
"""
function lvar_jac_param_coord(nlp::AbstractNLPModel{T, S}) where {T, S}
  vals = S(undef, get_nnzjplvar(nlp.meta))
  return lvar_jac_param_coord!(nlp, vals)
end

"""
    Jv = lvar_jpprod(nlp, v)

Evaluate `∇ₚlvar ⋅ v`, the lower variable-bound Jacobian product.
"""
function lvar_jpprod(nlp::AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
  @lencheck get_nparam(nlp.meta) v
  Jv = S(undef, get_nvar(nlp.meta))
  return lvar_jpprod!(nlp, v, Jv)
end

function lvar_jpprod! end

"""
    Jtv = lvar_jptprod(nlp, v)

Evaluate `∇ₚlvarᵀ ⋅ v`, the lower variable-bound Jacobian transpose product.
"""
function lvar_jptprod(nlp::AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
  @lencheck get_nvar(nlp.meta) v
  Jtv = S(undef, get_nparam(nlp.meta))
  return lvar_jptprod!(nlp, v, Jtv)
end

function lvar_jptprod! end

"""
    (rows, cols) = uvar_jac_param_structure(nlp)

Return the structure of the variable upper bound Jacobian wrt parameters in sparse coordinate format.
"""
function uvar_jac_param_structure(nlp::AbstractNLPModel)
  rows = Vector{Int}(undef, get_nnzjpuvar(nlp.meta))
  cols = Vector{Int}(undef, get_nnzjpuvar(nlp.meta))
  uvar_jac_param_structure!(nlp, rows, cols)
end

function uvar_jac_param_structure! end

"""
    vals = uvar_jac_param_coord!(nlp, vals)

Evaluate the upper variable-bound Jacobian wrt parameters in sparse coordinate format, overwriting `vals`.
"""
function uvar_jac_param_coord! end

"""
    vals = uvar_jac_param_coord(nlp)

Evaluate the upper variable-bound Jacobian wrt parameters in sparse coordinate format.
"""
function uvar_jac_param_coord(nlp::AbstractNLPModel{T, S}) where {T, S}
  vals = S(undef, get_nnzjpuvar(nlp.meta))
  return uvar_jac_param_coord!(nlp, vals)
end

"""
    Jv = uvar_jpprod(nlp, v)

Evaluate `∇ₚuvar ⋅ v`, the upper variable-bound Jacobian product.
"""
function uvar_jpprod(nlp::AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
  @lencheck get_nparam(nlp.meta) v
  Jv = S(undef, get_nvar(nlp.meta))
  return uvar_jpprod!(nlp, v, Jv)
end

function uvar_jpprod! end

"""
    Jtv = uvar_jptprod(nlp, v)

Evaluate `∇ₚuvarᵀ ⋅ v`, the upper variable-bound Jacobian transpose product.
"""
function uvar_jptprod(nlp::AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
  @lencheck get_nvar(nlp.meta) v
  Jtv = S(undef, get_nparam(nlp.meta))
  return uvar_jptprod!(nlp, v, Jtv)
end

function uvar_jptprod! end
