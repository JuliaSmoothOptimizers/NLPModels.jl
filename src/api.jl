export obj, grad, grad!, objgrad, objgrad!, objcons, objcons!, cons,
       cons!, jth_con, jth_congrad, jth_congrad!, jth_sparse_congrad,
       jac_coord, jac, jprod, jprod!, jtprod, jtprod!, jac_op, jac_op!,
       jth_hprod, jth_hprod!, ghjvprod, ghjvprod!, hess_coord, hess,
       hprod, hprod!, hess_op, hess_op!, chess, residual, residual!,
       jac_residual, jprod_residual, jprod_residual!, jtprod_residual,
       jtprod_residual!, jac_op_residual, jac_op_residual!,
       hess_residual, hprod_residual, hprod_residual!, hess_op_residual,
       hess_op_residual!, varscale, lagscale, conscale, push!

"""`f = obj(nlp, x)`

Evaluate \$f(x)\$, the objective function of `nlp` at `x`.
"""
function obj(nlp::AbstractNLPModel, x::AbstractVector)
  increment!(nlp, :neval_obj)
  # Defaults the objective funcions of empty problems to 0
  fx = 0.0
  for i = 1:nobjs(nlp)
    fx += nlp.σfs[i] * obj(nlp, i, x)
  end
  if nlsequ(nlp) > 0
    Fx = residual(nlp, x)
    fx += nlp.σnls * dot(Fx, Fx) / 2
  end
  if llsrows(nlp) > 0
    Fx = nlp.A * x - nlp.b
    fx += nlp.σls * dot(Fx, Fx) / 2
  end
  return fx
end

"""`f = obj(nlp, i, x)`

Evaluate \$f_i(x)\$, the i-th single objective function of `nlp` at `x`.
"""
obj(::AbstractNLPModel, ::Int, ::AbstractVector) =
  throw(NotImplementedError("obj"))

"""`g = grad(nlp, x)`

Evaluate \$\\nabla f(x)\$, the gradient of the objective function at `x`.
"""
function grad(nlp::AbstractNLPModel, x::AbstractVector)
  gx = zeros(nvar(nlp))
  return grad!(nlp, x, gx)
end

"""`g = grad(nlp, i, x)`

Evaluate \$\\nabla f_i(x)\$, the gradient of the i-th single objective
function at `x`.
"""
grad(::AbstractNLPModel, ::Int, ::AbstractVector) =
  throw(NotImplementedError("grad"))

"""`g = grad!(nlp, x, g)`

Evaluate \$\\nabla f(x)\$, the gradient of the objective function at `x`
in place.
"""
function grad!(nlp::AbstractNLPModel, x::AbstractVector, g::AbstractVector)
  increment!(nlp, :neval_grad)
  fill!(g, 0.0)
  if nobjs(nlp) > 0
    grad!(nlp, 1, x, g)
    nlp.σfs[1] != 1.0 && scale!(gx, nlp.σfs[1])
    if nobjs(nlp) > 1
      gx = zeros(nvar(nlp))
      for i = 2:nobjs(nlp)
        grad!(nlp, i, x, gx)
        g .+= nlp.σfs[i] .* gx
      end
    end
  end
  if nlsequ(nlp) > 0
    Fx = residual(nlp, x)
    if nobjs(nlp) == 0
      jtprod_residual!(nlp, x, Fx, g)
      g .= nlp.σnls * g
    else
      g .+= nlp.σnls .* jtprod_residual(nlp, x, Fx)
    end
  end
  if llsrows(nlp) > 0
    g .+= nlp.σls .* (nlp.A' * (nlp.A * x - nlp.b))
  end
  return g
end

"""`g = grad!(nlp, i, x, g)`

Evaluate \$\\nabla f_i(x)\$, the gradient of the i-th single objective
function at `x` in place.
"""
grad!(::AbstractNLPModel, ::Int, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("grad!"))

"""`c = cons(nlp, x)`

Evaluate \$c(x)\$, the constraints at `x`.
"""
cons(::AbstractNLPModel, ::AbstractVector) =
  throw(NotImplementedError("cons"))

"""`c = cons!(nlp, x, c)`

Evaluate \$c(x)\$, the constraints at `x` in place.
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

Evaluate \$f(x)\$ and \$c(x)\$ at `x`.
"""
function objcons(nlp, x)
  f = obj(nlp, x)
  c = ncon(nlp) > 0 ? cons(nlp, x) : []
  return f, c
end

"""`f = objcons!(nlp, x, c)`

Evaluate \$f(x)\$ and \$c(x)\$ at `x`. `c` is overwritten with the value of \$c(x)\$.
"""
function objcons!(nlp, x, c)
  f = obj(nlp, x)
  ncon(nlp) > 0 && cons!(nlp, x, c)
  return f, c
end

"""`f, g = objgrad(nlp, x)`

Evaluate \$f(x)\$ and \$\\nabla f(x)\$ at `x`.
"""
function objgrad(nlp, x)
  f = obj(nlp, x)
  g = grad(nlp, x)
  return f, g
end

"""`f, g = objgrad!(nlp, x, g)`

Evaluate \$f(x)\$ and \$\\nabla f(x)\$ at `x`. `g` is overwritten with the
value of \$\\nabla f(x)\$.
"""
function objgrad!(nlp, x, g)
  f = obj(nlp, x)
  grad!(nlp, x, g)
  return f, g
end

"""`(rows,cols,vals) = jac_coord(nlp, x)`

Evaluate \$\\nabla c(x)\$, the constraint's Jacobian at `x` in sparse coordinate format.
"""
jac_coord(::AbstractNLPModel, ::AbstractVector) =
  throw(NotImplementedError("jac_coord"))

"""`Jx = jac(nlp, x)`

Evaluate \$\\nabla c(x)\$, the constraint's Jacobian at `x` as a sparse matrix.
"""
jac(::AbstractNLPModel, ::AbstractVector) = throw(NotImplementedError("jac"))

"""`Jv = jprod(nlp, x, v)`

Evaluate \$\\nabla c(x)v\$, the Jacobian-vector product at `x`.
"""
jprod(::AbstractNLPModel, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("jprod"))

"""`Jv = jprod!(nlp, x, v, Jv)`

Evaluate \$\\nabla c(x)v\$, the Jacobian-vector product at `x` in place.
"""
jprod!(::AbstractNLPModel, ::AbstractVector, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("jprod!"))

"""`Jtv = jtprod(nlp, x, v, Jtv)`

Evaluate \$\\nabla c(x)^Tv\$, the transposed-Jacobian-vector product at `x`.
"""
jtprod(::AbstractNLPModel, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("jtprod"))

"""`Jtv = jtprod!(nlp, x, v, Jtv)`

Evaluate \$\\nabla c(x)^Tv\$, the transposed-Jacobian-vector product at `x` in place.
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
  return LinearOperator{Float64,F1,Nothing,F3}(ncon(nlp), nvar(nlp),
                                               false, false, prod, nothing, ctprod)
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
  return LinearOperator{Float64,F1,Nothing,F3}(ncon(nlp), nvar(nlp),
                                               false, false, prod, nothing, ctprod)
end

jth_hprod(::AbstractNLPModel, ::AbstractVector, ::AbstractVector, ::Integer) =
  throw(NotImplementedError("jth_hprod"))
jth_hprod!(::AbstractNLPModel, ::AbstractVector, ::AbstractVector, ::Integer, ::AbstractVector) =
  throw(NotImplementedError("jth_hprod!"))
ghjvprod(::AbstractNLPModel, ::AbstractVector, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("ghjvprod"))
ghjvprod!(::AbstractNLPModel, ::AbstractVector, ::AbstractVector, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("ghjvprod!"))

"""`(rows,cols,vals) = hess_coord(nlp, x; obj_weight=1.0, y=zeros)`

Evaluate the Lagrangian Hessian at `(x,y)` in sparse coordinate format,
with objective function scaled by `obj_weight`, i.e.,

\\\\[ \\nabla^2L(x,y) = \\sigma * \\nabla^2 f(x) + \\sum_{i=1}^m y_i\\nabla^2 c_i(x), \\\\]

with σ = obj_weight.
Only the lower triangle is returned.
"""
function hess_coord(nlp::AbstractNLPModel, x::AbstractVector;
                    obj_weight = 1.0, y :: AbstractVector = [])
  return findnz(hess(nlp, x, obj_weight=obj_weight, y=y))
end

"""`(rows,cols,vals) = hess_coord(nlp, i, x)`

Evaluate the Hessian of the i-th single objective function at `x` in
sparse coordinate format.
Only the lower triangle is returned.
"""
hess_coord(::AbstractNLPModel, ::Int, ::AbstractVector) =
  throw(NotImplementedError("hess_coord"))

"""`Hx = hess(nlp, x; obj_weight=1.0, y=zeros)`

Evaluate the Lagrangian Hessian at `(x,y)` as a sparse matrix,
with objective function scaled by `obj_weight`, i.e.,

\\\\[ \\nabla^2L(x,y) = \\sigma * \\nabla^2 f(x) + \\sum_{i=1}^m y_i\\nabla^2 c_i(x), \\\\]

with σ = obj_weight.
Only the lower triangle is returned.
"""
function hess(nlp::AbstractNLPModel, x::AbstractVector;
              obj_weight = 1.0, y :: AbstractVector = [])
  increment!(nlp, :neval_hess)
  Hx = spzeros(nvar(nlp), nvar(nlp))
  if obj_weight != 0.0
    for i = 1:nobjs(nlp)
      Hx .+= (obj_weight * nlp.σfs[i]) .* hess(nlp, i, x)
    end
    if nlsequ(nlp) > 0
      Jx = jac_residual(nlp, x)
      Hx .+= (obj_weight * nlp.σnls) * tril(Jx' * Jx)
      Fx = residual(nlp, x)
      m = length(Fx)
      for i = 1:m
        Hx .+= (obj_weight * nlp.σnls * Fx[i]) * hess_residual(nlp, x, i)
      end
    end
    if llsrows(nlp) > 0
      Hx .+= (obj_weight * nlp.σls) * (nlp.A' * nlp.A)
    end
  end
  for i = 1:min(length(y), ncon(nlp))
    if y[i] != 0.0
      Hx .+= y[i] * chess(nlp, i, x)
    end
  end
  return tril(Hx)
end

"""`Hx = hess(nlp, i, x)`

Evaluate the Hessian of the i-th single objective function at `x` as a
sparse matrix.
Only the lower triangle is returned.
"""
hess(::AbstractNLPModel, ::Int, ::AbstractVector) =
  throw(NotImplementedError("hess"))

"""`Cx = chess(nlp, i, x)`

Evaluate the Hessian of the i-th constraint function at `x` as a sparse
matrix.
Only the lower trinagle is returned.
"""
chess(::AbstractNLPModel, ::Int, ::AbstractVector) =
  throw(NotImplementedError("chess"))

"""`Hv = hprod(nlp, x, v; obj_weight=1.0, y=zeros)`

Evaluate the product of the Lagrangian Hessian at `(x,y)` with the vector `v`,
with objective function scaled by `obj_weight`, i.e.,

\\\\[ \\nabla^2L(x,y) = \\sigma * \\nabla^2 f(x) + \\sum_{i=1}^m y_i\\nabla^2 c_i(x), \\\\]

with σ = obj_weight.
"""
function hprod(nlp::AbstractNLPModel, x::AbstractVector, v::AbstractVector;
               obj_weight=1.0, y::AbstractVector=[])
  Hv = zeros(nvar(nlp))
  return hprod!(nlp, x, v, Hv, obj_weight=obj_weight, y=y)
end

"""`Hv = hprod(nlp, i, x, v)`

Evaluate the product of the Hessian of the i-th single objective
function at `x` with the vector `v`.
"""
hprod(::AbstractNLPModel, ::Int, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("hprod"))

"""`Hv = hprod!(nlp, x, v, Hv; obj_weight=1.0, y=zeros)`

Evaluate the product of the Lagrangian Hessian at `(x,y)` with the vector `v` in
place, with objective function scaled by `obj_weight`, i.e.,

\\\\[ \\nabla^2L(x,y) = \\sigma * \\nabla^2 f(x) + \\sum_{i=1}^m y_i\\nabla^2 c_i(x), \\\\]

with σ = obj_weight.
"""
function hprod!(nlp::AbstractNLPModel, x::AbstractVector,
                v::AbstractVector, Hv::AbstractVector;
                obj_weight=1.0, y::AbstractVector = [])
  increment!(nlp, :neval_hprod)
  n = nvar(nlp)
  fill!(Hv, 0.0)
  if obj_weight != 0.0
    Hiv = zeros(n)
    for i = 1:nobjs(nlp)
      hprod!(nlp, i, x, v, Hiv)
      Hv[1:n] .+= (obj_weight * nlp.σfs[i]) * Hiv
    end
    if nlsequ(nlp) > 0
      Jv = jprod_residual(nlp, x, v)
      Hv[1:n] .+= (obj_weight * nlp.σnls) * jtprod_residual(nlp, x, Jv)
      Fx = residual(nlp, x)
      m = length(Fx)
      Hiv = zeros(n)
      for i = 1:m
        hprod_residual!(nlp, x, i, v, Hiv)
        Hv[1:n] .+= (obj_weight * nlp.σnls * Fx[i]) * Hiv
      end
    end
    if llsrows(nlp) > 0
      Hv[1:n] .+= (obj_weight * nlp.σls) * (nlp.A' * (nlp.A * v))
    end
  end
  m = min(length(y), ncon(nlp))
  if m > 0
    Cv = zeros(n)
    for i = 1:m
      jth_hprod!(nlp, x, v, i, Cv)
      Hv[1:n] .+= y[i] * Cv
    end
  end
  return Hv
end

"""`Hv = hprod!(nlp, i, x, v, Hv)`

Evaluate the product of the Hessian of the i-th single objective
function at `x` with the vector `v` in place.
"""
hprod!(::AbstractNLPModel, ::Int, ::AbstractVector, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("hprod!"))

"""`H = hess_op(nlp, x; obj_weight=1.0, y=zeros)`

Return the Lagrangian Hessian at `(x,y)` with objective function scaled by
`obj_weight` as a linear operator. The resulting object may be used as if it were a
matrix, e.g., `H * v`. The linear operator H represents

\\\\[ \\nabla^2L(x,y) = \\sigma * \\nabla^2 f(x) + \\sum_{i=1}^m y_i\\nabla^2 c_i(x), \\\\]

with σ = obj_weight.
"""
function hess_op(nlp :: AbstractNLPModel, x :: AbstractVector;
                 obj_weight :: Float64=1.0, y :: AbstractVector=zeros(ncon(nlp)))
  prod = @closure v -> hprod(nlp, x, v; obj_weight=obj_weight, y=y)
  F = typeof(prod)
  return LinearOperator{Float64,F,Nothing,Nothing}(nvar(nlp), nvar(nlp),
                                                   true, true, prod, nothing, nothing)
end

"""`H = hess_op!(nlp, x, Hv; obj_weight=1.0, y=zeros)`

Return the Lagrangian Hessian at `(x,y)` with objective function scaled by
`obj_weight` as a linear operator, and storing the result on `Hv`. The resulting
object may be used as if it were a matrix, e.g., `w = H * v`. The vector `Hv` is
used as preallocated storage for the operation.  The linear operator H
represents

\\\\[ \\nabla^2L(x,y) = \\sigma * \\nabla^2 f(x) + \\sum_{i=1}^m y_i\\nabla^2 c_i(x), \\\\]

with σ = obj_weight.
"""
function hess_op!(nlp :: AbstractNLPModel, x :: AbstractVector, Hv :: AbstractVector;
                 obj_weight :: Float64=1.0, y :: AbstractVector=zeros(ncon(nlp)))
  prod = @closure v -> hprod!(nlp, x, v, Hv; obj_weight=obj_weight, y=y)
  F = typeof(prod)
  return LinearOperator{Float64,F,Nothing,Nothing}(nvar(nlp), nvar(nlp),
                                                   true, true, prod, nothing, nothing)
end

"""`H = hess_op(nlp, i, x)`

Return the Hessian of the i-th single objective function at `x`.  The
resulting object may be used as if it were a matrix, e.g., `H * v`.
"""
function hess_op(nlp :: AbstractNLPModel, i :: Int, x :: AbstractVector)
  prod = @closure v -> hprod(nlp, i, x, v)
  F = typeof(prod)
  return LinearOperator{Float64,F,Nothing,Nothing}(nvar(nlp), nvar(nlp),
                                                   true, true, prod, nothing, nothing)
end

"""`H = hess_op!(nlp, i, x, Hv)`

Return the Hessian of the i-th single objective function at `x` as a
linear operator, and storing the result on `Hv`. The resulting object
may be used as if it were a matrix, e.g., `w = H * v`. The vector `Hv`
is used as preallocated storage for the operation.
"""
function hess_op!(nlp :: AbstractNLPModel, i :: Int, x :: AbstractVector,
                  Hv :: AbstractVector)
  prod = @closure v -> hprod!(nlp, i, x, v, Hv)
  F = typeof(prod)
  return LinearOperator{Float64,F,Nothing,Nothing}(nvar(nlp), nvar(nlp),
                                                   true, true, prod, nothing, nothing)
end

"""
    Fx = residual(nlp, x)

Computes F(x), the residual at x.
"""
function residual(nlp :: AbstractNLPModel, x :: AbstractVector)
  Fx = zeros(nlsequ(nlp) + llsrows(nlp))
  residual!(nlp, x, Fx)
end

"""
    Fx = residual!(nlp, x, Fx)

Computes F(x), the residual at x.
"""
residual!(::AbstractNLPModel, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("residual!"))

"""
    Jx = jac_residual(nlp, x)

Computes J(x), the Jacobian of the residual at x.
"""
jac_residual(::AbstractNLPModel, ::AbstractVector) =
  throw(NotImplementedError("jac_residual"))

"""
    Jv = jprod_residual(nlp, x, v)

Computes the product of the Jacobian of the residual at x and a vector, i.e.,  J(x)*v.
"""
function jprod_residual(nlp :: AbstractNLPModel, x :: AbstractVector, v :: AbstractVector)
  Jv = zeros(nlsequ(nlp) + llsrows(nlp))
  jprod_residual!(nlp, x, v, Jv)
end

"""
    Jv = jprod_residual!(nlp, x, v, Jv)

Computes the product of the Jacobian of the residual at x and a vector, i.e.,  J(x)*v, storing it in `Jv`.
"""
jprod_residual!(::AbstractNLPModel, ::AbstractVector, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("jprod_residual!"))

"""
    Jtv = jtprod_residual(nlp, x, v)

Computes the product of the transpose of the Jacobian of the residual at x and a vector, i.e.,  J(x)'*v.
"""
function jtprod_residual(nlp :: AbstractNLPModel, x :: AbstractVector, v :: AbstractVector)
  Jtv = zeros(nvar(nlp))
  jtprod_residual!(nlp, x, v, Jtv)
end

"""
    Jtv = jtprod_residual!(nlp, x, v, Jtv)

Computes the product of the transpose of the Jacobian of the residual at x and a vector, i.e.,  J(x)'*v, storing it in `Jtv`.
"""
jtprod_residual!(::AbstractNLPModel, ::AbstractVector, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("jtprod_residual!"))

"""
    Jx = jac_op_residual(nlp, x)

Computes J(x), the Jacobian of the residual at x, in linear operator form.
"""
function jac_op_residual(nlp :: AbstractNLPModel, x :: AbstractVector)
  prod = @closure v -> jprod_residual(nlp, x, v)
  ctprod = @closure v -> jtprod_residual(nlp, x, v)
  F1 = typeof(prod)
  F3 = typeof(ctprod)
  return LinearOperator{Float64,F1,Nothing,F3}(nlsequ(nlp) + llsrows(nlp), nvar(nlp),
                                               false, false, prod, nothing, ctprod)
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
  return LinearOperator{Float64,F1,Nothing,F3}(nlsequ(nlp) + llsrows(nlp), nvar(nlp),
                                               false, false, prod, nothing, ctprod)
end

"""
    Hi = hess_residual(nlp, x, i)

Computes the Hessian of the i-th residual at x.
"""
hess_residual(::AbstractNLPModel, ::AbstractVector, ::Int) =
  throw(NotImplementedError("hess_residual"))

"""
    Hiv = hprod_residual(nlp, x, i, v)

Computes the product of the Hessian of the i-th residual at x, times the vector v.
"""
function hprod_residual(nlp :: AbstractNLPModel, x :: AbstractVector, i :: Int, v :: AbstractVector)
  Hv = zeros(nvar(nlp))
  hprod_residual!(nlp, x, i, v, Hv)
end

"""
    Hiv = hprod_residual!(nlp, x, i, v, Hiv)

Computes the product of the Hessian of the i-th residual at x, times the vector v, and stores it in vector Hiv.
"""
hprod_residual!(::AbstractNLPModel, ::AbstractVector, ::Int, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("hprod_residual!"))

"""
    Hop = hess_op_residual(nlp, x, i)

Computes the Hessian of the i-th residual at x, in linear operator form.
"""
function hess_op_residual(nlp :: AbstractNLPModel, x :: AbstractVector, i :: Int)
  prod = @closure v -> hprod_residual(nlp, x, i, v)
  F = typeof(prod)
  return LinearOperator{Float64,F,Nothing,Nothing}(nvar(nlp), nvar(nlp),
                                                   true, true, prod, nothing, nothing)
end

"""
    Hop = hess_op_residual!(nlp, x, i, Hiv)

Computes the Hessian of the i-th residual at x, in linear operator form. The vector `Hiv` is used as preallocated storage for the operation.
"""
function hess_op_residual!(nlp :: AbstractNLPModel, x :: AbstractVector, i :: Int, Hiv :: AbstractVector)
  prod = @closure v -> hprod_residual!(nlp, x, i, v, Hiv)
  F = typeof(prod)
  return LinearOperator{Float64,F,Nothing,Nothing}(nvar(nlp), nvar(nlp),
                                                   true, true, prod, nothing, nothing)
end

import Base.push!
push!(nlp :: AbstractNLPModel, args...; kwargs...) =
  throw(NotImplementedError("push!"))
varscale(::AbstractNLPModel, ::AbstractVector) =
  throw(NotImplementedError("varscale"))
lagscale(::AbstractNLPModel, ::Float64) =
  throw(NotImplementedError("lagscale"))
conscale(::AbstractNLPModel, ::AbstractVector) =
  throw(NotImplementedError("conscale"))
