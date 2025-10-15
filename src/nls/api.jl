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
function residual end

"""
    Fx = residual!(nls, x, Fx)

Computes ``F(x)``, the residual at x.
"""
function residual! end

"""
    Jx = jac_residual(nls, x)

Computes ``J(x)``, the Jacobian of the residual at x.
"""
function jac_residual end

"""
    (rows,cols) = jac_structure_residual!(nls, rows, cols)

Returns the structure of the constraint's Jacobian in sparse coordinate format in place.
"""
function jac_structure_residual! end

"""
    (rows,cols) = jac_structure_residual(nls)

Returns the structure of the constraint's Jacobian in sparse coordinate format.
"""
function jac_structure_residual end

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
function jac_coord_residual end

"""
    Jv = jprod_residual(nls, x, v)

Computes the product of the Jacobian of the residual at x and a vector, i.e.,  ``J(x)v``.
"""
function jprod_residual end

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
function jprod_residual! end

"""
    Jtv = jtprod_residual(nls, x, v)

Computes the product of the transpose of the Jacobian of the residual at x and a vector, i.e.,  ``J(x)^Tv``.
"""
function jtprod_residual end

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
function jtprod_residual! end

"""
    Jx = jac_op_residual(nls, x)

Computes ``J(x)``, the Jacobian of the residual at x, in linear operator form.
"""
function jac_op_residual end

"""
    Jx = jac_op_residual!(nls, x, Jv, Jtv)

Computes ``J(x)``, the Jacobian of the residual at x, in linear operator form. The
vectors `Jv` and `Jtv` are used as preallocated storage for the operations.
"""
function jac_op_residual! end

"""
    Jx = jac_op_residual!(nls, rows, cols, vals, Jv, Jtv)

Computes ``J(x)``, the Jacobian of the residual given by `(rows, cols, vals)`, in linear operator form. The
vectors `Jv` and `Jtv` are used as preallocated storage for the operations.
"""
function jac_op_residual! end

"""
    H = hess_residual(nls, x, v)

Computes the linear combination of the Hessians of the residuals at `x` with coefficients
`v`.
A `Symmetric` object wrapping the lower triangle is returned.
"""
function hess_residual end

"""
    (rows,cols) = hess_structure_residual(nls)

Returns the structure of the residual Hessian.
"""
function hess_structure_residual end

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
function hess_coord_residual end

"""
    Hj = jth_hess_residual(nls, x, j)

Computes the Hessian of the j-th residual at x.
"""
function jth_hess_residual end

"""
    vals = jth_hess_residual_coord(nls, x, j)

Evaluate the Hessian of j-th residual at `x` in sparse coordinate format.
Only the lower triangle is returned.
"""
function jth_hess_residual_coord end

"""
    vals = jth_hess_residual_coord!(nls, x, j, vals)

Evaluate the Hessian of j-th residual at `x` in sparse coordinate format, with `vals` of
length `nls.nls_meta.nnzh`, in place. Only the lower triangle is returned.
"""
function jth_hess_residual_coord! end

"""
    Hiv = hprod_residual(nls, x, i, v)

Computes the product of the Hessian of the i-th residual at x, times the vector v.
"""
function hprod_residual end

"""
    Hiv = hprod_residual!(nls, x, i, v, Hiv)

Computes the product of the Hessian of the i-th residual at x, times the vector v, and stores it in vector Hiv.
"""
function hprod_residual! end

"""
    Hop = hess_op_residual(nls, x, i)

Computes the Hessian of the i-th residual at x, in linear operator form.
"""
function hess_op_residual end

"""
    Hop = hess_op_residual!(nls, x, i, Hiv)

Computes the Hessian of the i-th residual at x, in linear operator form. The vector `Hiv` is used as preallocated storage for the operation.
"""
function hess_op_residual! end

"""
    f = obj(nls, x)
    f = obj(nls, x, Fx; recompute::Bool=true)

Evaluate `f(x)`, the objective function of `nls::AbstractNLSModel`. `Fx` is overwritten with the value of the residual `F(x)`.
If `recompute` is `true`, then `Fx` is updated with the residual at `x`.
"""
function obj end

function obj end

"""
    f, c = objcons!(nls, x, c)
    f, c = objcons!(nls, x, c, Fx; recompute::Bool=true)

In-place evaluation of constraints and objective for AbstractNLSModel.
`Fx` is overwritten with the value of the residual `F(x)`.
If `recompute` is `true`, then `Fx` is updated with the residual at `x`.
"""
function objcons! end

"""
    g = grad!(nls, x, g)
    g = grad!(nls, x, g, Fx; recompute::Bool=true)

Evaluate `∇f(x)`, the gradient of the objective function of `nls::AbstractNLSModel` at `x` in place. `Fx` is overwritten with the value of the residual `F(x)`.
If `recompute` is `true`, then `Fx` is updated with the residual at `x`.
"""
function grad! end

function grad! end

"""
    f, g = objgrad!(nls, x, g)
    f, g = objgrad!(nls, x, g, Fx; recompute::Bool=true)

Evaluate f(x) and ∇f(x) of `nls::AbstractNLSModel` at `x`. `Fx` is overwritten with the value of the residual `F(x)`.
If `recompute` is `true`, then `Fx` is updated with the residual at `x`.
"""
function objgrad! end
