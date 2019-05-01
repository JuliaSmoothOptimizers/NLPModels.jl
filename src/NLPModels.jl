__precompile__()

module NLPModels

# For documentation purpose
const LAGRANGIAN_HESSIAN = raw"""
```math
∇²L(x,y) = σ ∇²f(x) + ∑ᵢ yᵢ ∇²cᵢ(x),
```
with `σ = obj_weight`
"""

using LinearAlgebra, LinearOperators, Printf, SparseArrays, FastClosures

export AbstractNLPModelMeta, NLPModelMeta, AbstractNLPModel, Counters, reset!, sum_counters, push!,
       varscale, lagscale, conscale, NotImplementedError

# General objective
export general_obj, general_grad, general_grad!, general_hess, general_hprod, general_hprod!, general_hess_op, general_hess_op!,
       general_hess_structure, general_hess_structure!, general_hess_coord, general_hess_coord!

# Nonlinear least-squares
export residual, residual!, jac_residual, jac_coord_residual, jprod_residual, jprod_residual!, jtprod_residual,
       jtprod_residual!, jac_op_residual, jac_op_residual!, hess_residual, hess_structure_residual!,
       hess_structure_residual, hess_coord_residual!, hess_coord_residual, jth_hess_residual,
       hprod_residual, hprod_residual!, hess_op_residual, hess_op_residual!

# Constraints
export cons, cons!, jth_con, jth_congrad, jth_congrad!, jth_sparse_congrad, jac_structure, jac_coord!,
       jac_coord, jac, jprod, jprod!, jtprod, jtprod!, jac_op, jac_op!

# Sum of objectives
export obj, grad, grad!, objgrad, objgrad!, objcons, objcons!, jth_hprod, jth_hprod!, ghjvprod, ghjvprod!,
       hess_structure!, hess_structure, hess_coord!, hess_coord, hess, hprod, hprod!, hess_op, hess_op!

# import methods we override
import Base.push!
import LinearOperators.reset!

include("nlp_utils.jl")
include("nlp_types.jl")

mutable struct NotImplementedError <: Exception
  name :: Union{Symbol,Function,String}
end

Base.showerror(io::IO, e::NotImplementedError) = print(io, e.name, " not implemented")

# Methods to be overridden in other packages.
"""
    f = general_obj(nlp, x)

Evaluate ``f₀(x)``, the general objective function of `nlp` at `x`.
"""
general_obj(::AbstractNLPModel, ::AbstractVector) =
  throw(NotImplementedError("general_obj"))

"""`f = obj(nlp, x)`

Evaluate ``f(x)``, the objective function of `nlp` at `x`.
"""
function obj(nlp::AbstractNLPModel, x::AbstractVector)
  increment!(nlp, :neval_obj)
  fx = zero(eltype(x))
  if nlp.meta.has_general_obj
    fx += nlp.σf * general_obj(nlp, x)
  end
  if nlp.meta.num_residuals > 0
    Fx = residual(nlp, x)
    fx += nlp.σnls * dot(Fx, Fx) / 2
  end
  return fx
end

"""
    g = general_grad(nlp, x)

Evaluate ``∇f₀(x)``, the gradient of the general objective function at `x`.
"""
function general_grad(nlp::AbstractNLPModel, x::AbstractVector)
  g = similar(x)
  return general_grad!(nlp, x, g)
end

"""`g = grad(nlp, x)`

Evaluate ``∇f(x)``, the gradient of the objective function at `x`.
"""
function grad(nlp::AbstractNLPModel, x::AbstractVector)
  g = similar(x)
  return grad!(nlp, x, g)
end

"""
    g = general_grad!(nlp, x, g)

Evaluate ``∇f(x)``, the gradient of the general objective function at `x` in place.
"""
general_grad!(::AbstractNLPModel, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("general_grad!"))

"""`g = grad!(nlp, x, g)`

Evaluate ``∇f(x)``, the gradient of the objective function at `x` in place.
"""
function grad!(nlp::AbstractNLPModel, x::AbstractVector, g::AbstractVector)
  increment!(nlp, :neval_grad)
  if nlp.meta.has_general_obj
    general_grad!(nlp, x, g)
    g .*= nlp.σf
  else
    fill!(g, zero(eltype(x)))
  end
  if nlp.meta.num_residuals > 0
    Fx = residual(nlp, x)
    g .+= jtprod_residual(nlp, x, Fx) * nlp.σnls
  end
  return g
end

"""`c = cons(nlp, x)`

Evaluate ``c(x)``, the constraints at `x`.
"""
function cons(nlp::AbstractNLPModel, x::AbstractVector)
  c = similar(x, nlp.meta.ncon)
  return cons!(nlp, x, c)
end

"""`c = cons!(nlp, x, c)`

Evaluate ``c(x)``, the constraints at `x` in place.
"""
cons!(::AbstractNLPModel, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("cons!"))

jth_con(::AbstractNLPModel, ::AbstractVector, ::Integer) =
  throw(NotImplementedError("jth_con"))
jth_congrad(::AbstractNLPModel, ::AbstractVector, ::Integer) =
  throw(NotImplementedError("jth_congrad"))
jth_congrad!(::AbstractNLPModel, ::AbstractVector, ::Integer, ::AbstractVector) =
  throw(NotImplementedError("jth_congrad!"))
jth_sparse_congrad(::AbstractNLPModel, ::AbstractVector, ::Integer) =
  throw(NotImplementedError("jth_sparse_congrad"))

"""`f, c = objcons(nlp, x)`

Evaluate ``f(x)`` and ``c(x)`` at `x`.
"""
function objcons(nlp, x)
  f = obj(nlp, x)
  c = nlp.meta.ncon > 0 ? cons(nlp, x) : Float64[]
  return f, c
end

"""`f = objcons!(nlp, x, c)`

Evaluate ``f(x)`` and ``c(x)`` at `x`. `c` is overwritten with the value of ``c(x)``.
"""
function objcons!(nlp, x, c)
  f = obj(nlp, x)
  nlp.meta.ncon > 0 && cons!(nlp, x, c)
  return f, c
end

"""`f, g = objgrad(nlp, x)`

Evaluate ``f(x)`` and ``∇f(x)`` at `x`.
"""
function objgrad(nlp, x)
  f = obj(nlp, x)
  g = grad(nlp, x)
  return f, g
end

"""`f, g = objgrad!(nlp, x, g)`

Evaluate ``f(x)`` and ``∇f(x)`` at `x`. `g` is overwritten with the
value of ``∇f(x)``.
"""
function objgrad!(nlp, x, g)
  f = obj(nlp, x)
  grad!(nlp, x, g)
  return f, g
end

"""`(rows,cols) = jac_structure(nlp)`

Returns the structure of the constraint's Jacobian in sparse coordinate format.
"""
jac_structure(:: AbstractNLPModel) = throw(NotImplementedError("jac_structure"))

"""`(rows,cols,vals) = jac_coord!(nlp, x, rows, cols, vals)`

Evaluate ``∇c(x)``, the constraint's Jacobian at `x` in sparse coordinate format,
rewriting `vals`. `rows` and `cols` are not rewritten.
"""
jac_coord!(:: AbstractNLPModel, :: AbstractVector) = throw(NotImplementedError("jac_coord!"))

"""`(rows,cols,vals) = jac_coord(nlp, x)`

Evaluate ``∇c(x)``, the constraint's Jacobian at `x` in sparse coordinate format.
"""
jac_coord(:: AbstractNLPModel, :: AbstractVector) = throw(NotImplementedError("jac_coord"))

"""`Jx = jac(nlp, x)`

Evaluate ``∇c(x)``, the constraint's Jacobian at `x` as a sparse matrix.
"""
jac(::AbstractNLPModel, ::AbstractVector) = throw(NotImplementedError("jac"))

"""`Jv = jprod(nlp, x, v)`

Evaluate ``∇c(x)v``, the Jacobian-vector product at `x`.
"""
function jprod(nlp::AbstractNLPModel, x::AbstractVector, v::AbstractVector)
  Jv = similar(v, nlp.meta.ncon)
  return jprod!(nlp, x, v, Jv)
end

"""`Jv = jprod!(nlp, x, v, Jv)`

Evaluate ``∇c(x)v``, the Jacobian-vector product at `x` in place.
"""
jprod!(::AbstractNLPModel, ::AbstractVector, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("jprod!"))

"""`Jtv = jtprod(nlp, x, v, Jtv)`

Evaluate ``∇c(x)^Tv``, the transposed-Jacobian-vector product at `x`.
"""
function jtprod(nlp::AbstractNLPModel, x::AbstractVector, v::AbstractVector)
  Jtv = similar(x)
  return jtprod!(nlp, x, v, Jtv)
end

"""`Jtv = jtprod!(nlp, x, v, Jtv)`

Evaluate ``∇c(x)^Tv``, the transposed-Jacobian-vector product at `x` in place.
"""
jtprod!(::AbstractNLPModel, ::AbstractVector, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("jtprod!"))

"""`J = jac_op(nlp, x)`

Return the Jacobian at `x` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or
`J' * v`.
"""
function jac_op(nlp :: AbstractNLPModel, x :: AbstractVector)
  prod = @closure v -> jprod(nlp, x, v)
  ctprod = @closure v -> jtprod(nlp, x, v)
  F1 = typeof(prod)
  F3 = typeof(ctprod)
  return LinearOperator{Float64,F1,F3,F3}(nlp.meta.ncon, nlp.meta.nvar,
                                          false, false, prod, ctprod, ctprod)
end

"""`J = jac_op!(nlp, x, Jv, Jtv)`

Return the Jacobian at `x` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or
`J' * v`. The values `Jv` and `Jtv` are used as preallocated storage for the
operations.
"""
function jac_op!(nlp :: AbstractNLPModel, x :: AbstractVector,
                 Jv :: AbstractVector, Jtv :: AbstractVector)
  prod = @closure v -> jprod!(nlp, x, v, Jv)
  ctprod = @closure v -> jtprod!(nlp, x, v, Jtv)
  F1 = typeof(prod)
  F3 = typeof(ctprod)
  return LinearOperator{Float64,F1,F3,F3}(nlp.meta.ncon, nlp.meta.nvar,
                                          false, false, prod, ctprod, ctprod)
end

jth_hprod(::AbstractNLPModel, ::AbstractVector, ::AbstractVector, ::Integer) =
  throw(NotImplementedError("jth_hprod"))
jth_hprod!(::AbstractNLPModel, ::AbstractVector, ::AbstractVector, ::Integer, ::AbstractVector) =
  throw(NotImplementedError("jth_hprod!"))
ghjvprod(::AbstractNLPModel, ::AbstractVector, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("ghjvprod"))
ghjvprod!(::AbstractNLPModel, ::AbstractVector, ::AbstractVector, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("ghjvprod!"))

"""
    (rows,cols) = general_hess_structure_general(nlp)

Returns the structure of the general objective Hessian in sparse coordinate format.
"""
function general_hess_structure(nlp :: AbstractNLPModel)
  rows = Vector{Int}(undef, nlp.meta.nnzh)
  cols = Vector{Int}(undef, nlp.meta.nnzh)
  general_hess_structure!(nlp, rows, cols)
end

"""`(rows,cols) = hess_structure(nlp)`

Returns the structure of the Lagrangian Hessian in sparse coordinate format.
"""
function hess_structure(nlp :: AbstractNLPModel)
  rows = Vector{Int}(undef, nlp.meta.nnzh)
  cols = Vector{Int}(undef, nlp.meta.nnzh)
  hess_structure!(nlp, rows, cols)
end

"""
    general_hess_structure!(nlp, rows, cols)

Returns the structure of the general objective Hessian in sparse coordinate format in place.
"""
general_hess_structure!(:: AbstractNLPModel, ::AbstractVector{<: Integer}, ::AbstractVector{<: Integer}) = throw(NotImplementedError("general_hess_structure!"))

"""`hess_structure!(nlp, rows, cols)`

Returns the structure of the Lagrangian Hessian in sparse coordinate format in place.
"""
hess_structure!(:: AbstractNLPModel, ::AbstractVector{<: Integer}, ::AbstractVector{<: Integer}) = throw(NotImplementedError("hess_structure!"))

"""
    (rows,cols,vals) = general_hess_coord!(nlp, x, rows, cols, vals)

Evaluate the general objective Hessian at `(x,y)` in sparse coordinate format, rewriting `vals`.
`rows` and `cols` are not rewritten. Only the lower triangle is returned.
"""
general_hess_coord!(:: AbstractNLPModel, :: AbstractVector, ::AbstractVector{<: Integer}, ::AbstractVector{<: Integer}, ::AbstractVector) = throw(NotImplementedError("general_hess_coord!"))

"""`(rows,cols,vals) = hess_coord!(nlp, x, rows, cols, vals; obj_weight=1.0, y=zeros)`

Evaluate the Lagrangian Hessian at `(x,y)` in sparse coordinate format,
with objective function scaled by `obj_weight`, i.e.,
$(LAGRANGIAN_HESSIAN),rewriting `vals`. `rows` and `cols` are not rewritten.
Only the lower triangle is returned.
"""
hess_coord!(:: AbstractNLPModel, :: AbstractVector, ::AbstractVector{<: Integer}, ::AbstractVector{<: Integer}, ::AbstractVector; kwargs...) = throw(NotImplementedError("hess_coord!"))

"""
    (rows,cols,vals) = general_hess_coord(nlp, x; obj_weight=1.0, y=zeros)

Evaluate the general objective Hessian at `(x,y)` in sparse coordinate format.
Only the lower triangle is returned.
"""
general_hess_coord(nlp::AbstractNLPModel, x::AbstractVector; y::AbstractVector=Float64[], obj_weight::Real=one(eltype(x))) = throw(NotImplementedError("general_hess_coord"))


"""`(rows,cols,vals) = hess_coord(nlp, x; obj_weight=1.0, y=zeros)`

Evaluate the Lagrangian Hessian at `(x,y)` in sparse coordinate format,
with objective function scaled by `obj_weight`, i.e.,

$(LAGRANGIAN_HESSIAN).
Only the lower triangle is returned.
"""
hess_coord(nlp::AbstractNLPModel, x::AbstractVector; y::AbstractVector=eltype(x)[], obj_weight::Real=one(eltype(x))) = throw(NotImplementedError("hess_coord"))

"""
    Hx = general_hess(nlp, x)

Evaluate the general objective Hessian at `(x,y)` as a sparse matrix.
Only the lower triangle is returned.
"""
general_hess(::AbstractNLPModel, ::AbstractVector) =
  throw(NotImplementedError("general_hess"))

"""`Hx = hess(nlp, x; obj_weight=1.0, y=zeros)`

Evaluate the Lagrangian Hessian at `(x,y)` as a sparse matrix,
with objective function scaled by `obj_weight`, i.e.,

$(LAGRANGIAN_HESSIAN).
Only the lower triangle is returned.
"""
hess(::AbstractNLPModel, ::AbstractVector; kwargs...) =
  throw(NotImplementedError("hess"))

"""
    Hv = general_hprod(nlp, x, v)

Evaluate the product of the general objective Hessian at `(x,y)` with the vector `v`.
"""
function general_hprod(nlp::AbstractNLPModel, x::AbstractVector, v::AbstractVector)
  Hv = similar(x)
  return general_hprod!(nlp, x, v, Hv)
end

"""`Hv = hprod(nlp, x, v; obj_weight=1.0, y=zeros)`

Evaluate the product of the Lagrangian Hessian at `(x,y)` with the vector `v`,
with objective function scaled by `obj_weight`, where the Lagrangian Hessian is
$(LAGRANGIAN_HESSIAN).
"""
function hprod(nlp::AbstractNLPModel, x::AbstractVector, v::AbstractVector; obj_weight::Real = one(eltype(x)), y::AbstractVector=similar(x, 0))
  Hv = similar(x)
  return hprod!(nlp, x, v, Hv, obj_weight=obj_weight, y=y)
end

"""
    Hv = general_hprod!(nlp, x, v, Hv)

Evaluate the product of the general objective Hessian at `(x,y)` with the vector `v` in
place.
"""
general_hprod!(::AbstractNLPModel, ::AbstractVector, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("general_hprod!"))

"""`Hv = hprod!(nlp, x, v, Hv; obj_weight=1.0, y=zeros)`

Evaluate the product of the Lagrangian Hessian at `(x,y)` with the vector `v` in
place, with objective function scaled by `obj_weight`, where the Lagrangian Hessian is
$(LAGRANGIAN_HESSIAN).
"""
function hprod!(nlp::AbstractNLPModel, x::AbstractVector, v::AbstractVector, Hv::AbstractVector; obj_weight::Real = one(eltype(x)), y::AbstractVector=similar(x, 0))
  throw(NotImplementedError("hprod!"))
end

"""
    H = general_hess_op(nlp, x)

Return the general objective Hessian at `(x,y)`. The resulting object may be used as if
it were a matrix, e.g., `H * v`.
"""
function general_hess_op(nlp :: AbstractNLPModel, x :: AbstractVector)
  prod = @closure v -> general_hprod(nlp, x, v)
  F = typeof(prod)
  return LinearOperator{Float64,F,F,F}(nlp.meta.nvar, nlp.meta.nvar,
                                       true, true, prod, prod, prod)
end

"""`H = hess_op(nlp, x; obj_weight=1.0, y=zeros)`

Return the Lagrangian Hessian at `(x,y)` with objective function scaled by
`obj_weight` as a linear operator. The resulting object may be used as if it were a
matrix, e.g., `H * v`. The linear operator H represents
$(LAGRANGIAN_HESSIAN).
"""
function hess_op(nlp :: AbstractNLPModel, x :: AbstractVector;
                 obj_weight :: Float64=1.0, y :: AbstractVector=zeros(nlp.meta.ncon))
  prod = @closure v -> hprod(nlp, x, v; obj_weight=obj_weight, y=y)
  F = typeof(prod)
  return LinearOperator{Float64,F,F,F}(nlp.meta.nvar, nlp.meta.nvar,
                                       true, true, prod, prod, prod)
end

"""
    H = general_hess_op!(nlp, x, Hv)

Return the general objective Hessian at `(x,y)` storing the result on `Hv`. The resulting
object may be used as if it were a matrix, e.g., `w = H * v`. The vector `Hv` is
used as preallocated storage for the operation.
"""
function general_hess_op!(nlp :: AbstractNLPModel, x :: AbstractVector, Hv :: AbstractVector)
  prod = @closure v -> general_hprod!(nlp, x, v, Hv)
  F = typeof(prod)
  return LinearOperator{Float64,F,F,F}(nlp.meta.nvar, nlp.meta.nvar,
                                       true, true, prod, prod, prod)
end

"""`H = hess_op!(nlp, x, Hv; obj_weight=1.0, y=zeros)`

Return the Lagrangian Hessian at `(x,y)` with objective function scaled by
`obj_weight` as a linear operator, and storing the result on `Hv`. The resulting
object may be used as if it were a matrix, e.g., `w = H * v`. The vector `Hv` is
used as preallocated storage for the operation.  The linear operator H
represents
$(LAGRANGIAN_HESSIAN).
"""
function hess_op!(nlp :: AbstractNLPModel, x :: AbstractVector, Hv :: AbstractVector;
                 obj_weight :: Float64=1.0, y :: AbstractVector=zeros(nlp.meta.ncon))
  prod = @closure v -> hprod!(nlp, x, v, Hv; obj_weight=obj_weight, y=y)
  F = typeof(prod)
  return LinearOperator{Float64,F,F,F}(nlp.meta.nvar, nlp.meta.nvar,
                                       true, true, prod, prod, prod)
end

"""
    Fx = residual(nlp, x)

Computes F(x), the residual at x.
"""
function residual(nlp :: AbstractNLPModel, x :: AbstractVector)
  Fx = zeros(eltype(x), nlp.meta.num_residuals)
  residual!(nlp, x, Fx)
end

"""
    Fx = residual!(nlp, x, Fx)

Computes F(x), the residual at x.
"""
function residual!(nlp :: AbstractNLPModel, x :: AbstractVector, Fx :: AbstractVector)
  throw(NotImplementedError("residual!"))
end

"""
    Jx = jac_residual(nlp, x)

Computes J(x), the Jacobian of the residual at x.
"""
function jac_residual(nlp :: AbstractNLPModel, x :: AbstractVector)
  throw(NotImplementedError("jac_residual"))
end

"""
    (rows,cols,vals) = jac_coord_residual(nlp, x)

Computes the Jacobian of the residual at `x` in sparse coordinate format.
"""
function jac_coord_residual(nlp :: AbstractNLPModel, x :: AbstractVector)
  throw(NotImplementedError("jac_coord_residual"))
end

"""
    Jv = jprod_residual(nlp, x, v)

Computes the product of the Jacobian of the residual at x and a vector, i.e.,  J(x)*v.
"""
function jprod_residual(nlp :: AbstractNLPModel, x :: AbstractVector, v :: AbstractVector)
  Jv = zeros(eltype(x), nlp.meta.num_residuals)
  jprod_residual!(nlp, x, v, Jv)
end

"""
    Jv = jprod_residual!(nlp, x, v, Jv)

Computes the product of the Jacobian of the residual at x and a vector, i.e.,  J(x)*v, storing it in `Jv`.
"""
function jprod_residual!(nlp :: AbstractNLPModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  throw(NotImplementedError("jprod_residual!"))
end

"""
    Jtv = jtprod_residual(nlp, x, v)

Computes the product of the transpose of the Jacobian of the residual at x and a vector, i.e.,  J(x)'*v.
"""
function jtprod_residual(nlp :: AbstractNLPModel, x :: AbstractVector, v :: AbstractVector)
  Jtv = zeros(eltype(x), nlp.meta.nvar)
  jtprod_residual!(nlp, x, v, Jtv)
end

"""
    Jtv = jtprod_residual!(nlp, x, v, Jtv)

Computes the product of the transpose of the Jacobian of the residual at x and a vector, i.e.,  J(x)'*v, storing it in `Jtv`.
"""
function jtprod_residual!(nlp :: AbstractNLPModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  throw(NotImplementedError("jtprod_residual!"))
end

"""
    Jx = jac_op_residual(nlp, x)

Computes J(x), the Jacobian of the residual at x, in linear operator form.
"""
function jac_op_residual(nlp :: AbstractNLPModel, x :: AbstractVector)
  prod = @closure v -> jprod_residual(nlp, x, v)
  ctprod = @closure v -> jtprod_residual(nlp, x, v)
  F1 = typeof(prod)
  F3 = typeof(ctprod)
  return LinearOperator{Float64,F1,F3,F3}(nlp.meta.num_residuals, nlp.meta.nvar,
                                          false, false, prod, ctprod, ctprod)
end

"""
    Jx = jac_op_residual!(nlp, x, Jv, Jtv)

Computes J(x), the Jacobian of the residual at x, in linear operator form. The
vectors `Jv` and `Jtv` are used as preallocated storage for the operations.
"""
function jac_op_residual!(nlp :: AbstractNLPModel, x :: AbstractVector,
                          Jv :: AbstractVector, Jtv :: AbstractVector)
  prod = @closure v -> jprod_residual!(nlp, x, v, Jv)
  ctprod = @closure v -> jtprod_residual!(nlp, x, v, Jtv)
  F1 = typeof(prod)
  F3 = typeof(ctprod)
  return LinearOperator{Float64,F1,F3,F3}(nlp.meta.num_residuals, nlp.meta.nvar,
                                          false, false, prod, ctprod, ctprod)
end

"""
    H = hess_residual(nlp, x, v)

Computes the linear combination of the Hessians of the residuals at `x` with coefficients
`v`.
"""
function hess_residual(nlp :: AbstractNLPModel, x :: AbstractVector, v :: AbstractVector)
  throw(NotImplementedError("hess_residual"))
end

"""
    (rows,cols) = hess_structure_residual(nlp)

Returns the structure of the residual Hessian.
"""
function hess_structure_residual(nlp :: AbstractNLPModel)
  rows = Vector{Int}(undef, nlp.meta.nnzhF)
  cols = Vector{Int}(undef, nlp.meta.nnzhF)
  hess_structure_residual!(nlp, rows, cols)
end

"""
    hess_structure_residual!(nlp, rows, cols)

Returns the structure of the residual Hessian in place.
"""
function hess_structure_residual!(nlp :: AbstractNLPModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  throw(NotImplementedError("hess_structure_residual!"))
end

"""
    (rows,cols,vals) = hess_coord_residual!(nlp, x, v, rows, cols, vals)

Computes the linear combination of the Hessians of the residuals at `x` with coefficients
`v` in sparse coordinate format, rewriting `vals`.
"""
function hess_coord_residual!(nlp :: AbstractNLPModel, x :: AbstractVector, v :: AbstractVector, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer}, vals :: AbstractVector)
  throw(NotImplementedError("hess_coord_residual!"))
end

"""
    (rows,cols,vals) = hess_coord_residual(nlp, x, v)

Computes the linear combination of the Hessians of the residuals at `x` with coefficients
`v` in sparse coordinate format.
"""
function hess_coord_residual(nlp :: AbstractNLPModel, x :: AbstractVector, v :: AbstractVector)
  throw(NotImplementedError("hess_coord_residual"))
end

"""
    Hj = jth_hess_residual(nlp, x, j)

Computes the Hessian of the j-th residual at x.
"""
function jth_hess_residual(nlp :: AbstractNLPModel, x :: AbstractVector, i :: Int)
  throw(NotImplementedError("jth_hess_residual"))
end

"""
    Hiv = hprod_residual(nlp, x, i, v)

Computes the product of the Hessian of the i-th residual at x, times the vector v.
"""
function hprod_residual(nlp :: AbstractNLPModel, x :: AbstractVector, i :: Int, v :: AbstractVector)
  Hv = zeros(eltype(x), nlp.meta.nvar)
  hprod_residual!(nlp, x, i, v, Hv)
end

"""
    Hiv = hprod_residual!(nlp, x, i, v, Hiv)

Computes the product of the Hessian of the i-th residual at x, times the vector v, and stores it in vector Hiv.
"""
function hprod_residual!(nlp :: AbstractNLPModel, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  throw(NotImplementedError("hprod_residual!"))
end

"""
    Hop = hess_op_residual(nlp, x, i)

Computes the Hessian of the i-th residual at x, in linear operator form.
"""
function hess_op_residual(nlp :: AbstractNLPModel, x :: AbstractVector, i :: Int)
  prod = @closure v -> hprod_residual(nlp, x, i, v)
  F = typeof(prod)
  return LinearOperator{Float64,F,F,F}(nlp.meta.nvar, nlp.meta.nvar,
                                       true, true, prod, prod, prod)
end

"""
    Hop = hess_op_residual!(nlp, x, i, Hiv)

Computes the Hessian of the i-th residual at x, in linear operator form. The vector `Hiv` is used as preallocated storage for the operation.
"""
function hess_op_residual!(nlp :: AbstractNLPModel, x :: AbstractVector, i :: Int, Hiv :: AbstractVector)
  prod = @closure v -> hprod_residual!(nlp, x, i, v, Hiv)
  F = typeof(prod)
  return LinearOperator{Float64,F,F,F}(nlp.meta.nvar, nlp.meta.nvar,
                                       true, true, prod, prod, prod)
end

push!(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("push!"))
varscale(::AbstractNLPModel, ::AbstractVector) =
  throw(NotImplementedError("varscale"))
lagscale(::AbstractNLPModel, ::Float64) =
  throw(NotImplementedError("lagscale"))
conscale(::AbstractNLPModel, ::AbstractVector) =
  throw(NotImplementedError("conscale"))


include("autodiff_model.jl")
include("slack_model.jl")
include("qn_model.jl")
include("feasibility_residual.jl")
include("feasibility_form_nls.jl")
include("lls_model.jl")

include("dercheck.jl")

end # module
