# A simple derivative checker for AbstractNLPModels.
# D. Orban, March 2016.
# dominique.orban@gmail.com

export gradient_check, jacobian_check, hessian_check, hessian_check_from_grad


"""Check the first derivatives of the objective at `x` against centered
finite differences.

This function returns a dictionary indexed by components of the gradient for
which the relative error exceeds `rtol`.
"""
function gradient_check(nlp :: AbstractNLPModel;
                        x :: AbstractVector=nlp.meta.x0,
                        atol :: Float64=1.0e-6, rtol :: Float64=1.0e-4)

  # Optimal-ish step for second-order centered finite differences.
  step = (eps(Float64) / 3)^(1/3)

  # Check objective gradient.
  g_errs = Dict{Int, Float64}()
  g = grad(nlp, x)
  h = zeros(nlp.meta.nvar)
  for i = 1 : nlp.meta.nvar
    h[i] = step
    dfdxi = (obj(nlp, x + h) - obj(nlp, x - h)) / 2 / step
    err = abs(dfdxi - g[i])
    if err > atol + rtol * abs(dfdxi)
      g_errs[i] = err
    end
    h[i] = 0
  end
  return g_errs
end


"""Check the first derivatives of the constraints at `x` against centered
finite differences.

This function returns a dictionary indexed by (j, i) tuples such that the
relative error in the `i`-th partial derivative of the `j`-th constraint
exceeds `rtol`.
"""
function jacobian_check(nlp :: AbstractNLPModel;
                        x :: AbstractVector=nlp.meta.x0,
                        atol :: Float64=1.0e-6, rtol :: Float64=1.0e-4)

  # Fast exit if there are no constraints.
  J_errs = Dict{Tuple{Int,Int}, Float64}()
  nlp.meta.ncon > 0 || return J_errs

  # Optimal-ish step for second-order centered finite differences.
  step = (eps(Float64) / 3)^(1/3)

  # Check constraints Jacobian.
  J = jac(nlp, x)
  h = zeros(nlp.meta.nvar)
  cxph = zeros(nlp.meta.ncon)
  cxmh = zeros(nlp.meta.ncon)
  # Differentiate all constraints with respect to each variable in turn.
  for i = 1 : nlp.meta.nvar
    h[i] = step
    cons!(nlp, x + h, cxph)
    cons!(nlp, x - h, cxmh)
    dcdxi = (cxph - cxmh) / 2 / step
    for j = 1 : nlp.meta.ncon
      err = abs(dcdxi[j] - J[j, i])
      if err > atol + rtol * abs(dcdxi[j])
        J_errs[(j, i)] = err
      end
    end
    h[i] = 0
  end
  return J_errs
end


"""Check the second derivatives of the objective and each constraints at `x`
against centered finite differences. This check does not rely on exactness of
the first derivatives, only on objective and constraint values.

The `sgn` arguments refers to the formulation of the Lagrangian in the problem.
It should have a positive value if the Lagrangian is formulated as

    L(x,y) = f(x) + ∑ yⱼ cⱼ(x)

e.g., as in `JuMPNLPModel`s, and a negative value if the Lagrangian is
formulated as

    L(x,y) = f(x) - ∑ yⱼ cⱼ(x)

e.g., as in `AmplModel`s. Only the sign of `sgn` is important.

This function returns a dictionary indexed by functions. The 0-th function is
the objective while the k-th function (for k > 0) is the k-th constraint. The
values of the dictionary are dictionaries indexed by tuples (i, j) such that
the relative error in the second derivative ∂²fₖ/∂xᵢ∂xⱼ exceeds `rtol`.
"""
function hessian_check(nlp :: AbstractNLPModel;
                       x :: AbstractVector=nlp.meta.x0,
                       atol :: Float64=1.0e-6, rtol :: Float64=1.0e-4,
                       sgn :: Int=1)

  H_errs = Dict{Int, Dict{Tuple{Int,Int}, Float64}}()

  # Optimal-ish step for second-order centered finite differences.
  step = eps(Float64)^(1/4)
  sgn == 0 && error("sgn cannot be zero")
  sgn = sign(sgn)
  hi = zeros(nlp.meta.nvar)
  hj = zeros(nlp.meta.nvar)

  k = 0
  H_errs[k] = Dict{Tuple{Int,Int}, Float64}()
  H = hess(nlp, x)
  for i = 1 : nlp.meta.nvar
    hi[i] = step
    for j = 1 : i
      hj[j] = step
      d2fdxidxj = (obj(nlp, x + hi + hj) - obj(nlp, x - hi + hj) - obj(nlp, x + hi - hj) + obj(nlp, x - hi - hj)) / 4 / step^2
      err = abs(d2fdxidxj - H[i, j])
      if err > atol + rtol * abs(d2fdxidxj)
        H_errs[k][(i,j)] = err
      end
      hj[j] = 0
    end
    hi[i] = 0
  end

  y = zeros(nlp.meta.ncon)
  cxpp = zeros(nlp.meta.ncon)
  cxmp = zeros(nlp.meta.ncon)
  cxpm = zeros(nlp.meta.ncon)
  cxmm = zeros(nlp.meta.ncon)
  for k = 1 : nlp.meta.ncon
    H_errs[k] = Dict{Tuple{Int,Int}, Float64}()
    y[k] = sgn
    Hk = hess(nlp, x, obj_weight=0.0, y=y)
    for i = 1 : nlp.meta.nvar
      hi[i] = step
      for j = 1 : i
        hj[j] = step
        cons!(nlp, x + hi + hj, cxpp)
        cons!(nlp, x - hi + hj, cxmp)
        cons!(nlp, x + hi - hj, cxpm)
        cons!(nlp, x - hi - hj, cxmm)
        d2cdxidxj = (cxpp - cxmp - cxpm + cxmm) / 4 / step^2
        err = abs(d2cdxidxj[k] - Hk[i, j])
        if err > atol + rtol * abs(d2cdxidxj[k])
          println(d2cdxidxj[k], Hk[i, j])
          H_errs[k][(i,j)] = err
        end
        hj[j] = 0
      end
      hi[i] = 0
    end
    y[k] = 0
  end

  return H_errs
end


"""Check the second derivatives of the objective and each constraints at `x`
against centered finite differences. This check assumes exactness of the first
derivatives.

The `sgn` arguments refers to the formulation of the Lagrangian in the problem.
It should have a positive value if the Lagrangian is formulated as

    L(x,y) = f(x) + ∑ yⱼ cⱼ(x)

e.g., as in `JuMPNLPModel`s, and a negative value if the Lagrangian is
formulated as

    L(x,y) = f(x) - ∑ yⱼ cⱼ(x)

e.g., as in `AmplModel`s. Only the sign of `sgn` is important.

This function returns a dictionary indexed by functions. The 0-th function is
the objective while the k-th function (for k > 0) is the k-th constraint. The
values of the dictionary are dictionaries indexed by tuples (i, j) such that
the relative error in the second derivative ∂²fₖ/∂xᵢ∂xⱼ exceeds `rtol`.
"""
function hessian_check_from_grad(nlp :: AbstractNLPModel;
                                 x :: AbstractVector=nlp.meta.x0,
                                 atol :: Float64=1.0e-6, rtol :: Float64=1.0e-4,
                                 sgn :: Int=1)

  H_errs = Dict{Int, Dict{Tuple{Int,Int}, Float64}}()

  # Optimal-ish step for second-order centered finite differences.
  step = (eps(Float64) / 3)^(1/3)
  sgn == 0 && error("sgn cannot be zero")
  sgn = sign(sgn)
  h = zeros(nlp.meta.nvar)

  k = 0
  H_errs[k] = Dict{Tuple{Int,Int}, Float64}()
  H = hess(nlp, x)
  gxph = zeros(nlp.meta.nvar)
  gxmh = zeros(nlp.meta.nvar)
  for i = 1 : nlp.meta.nvar
    h[i] = step
    grad!(nlp, x + h, gxph)
    grad!(nlp, x - h, gxmh)
    dgdxi = (gxph - gxmh) / 2 / step
    for j = 1 : i
      err = abs(dgdxi[j] - H[i, j])
      if err > atol + rtol * abs(dgdxi[j])
        H_errs[k][(i,j)] = err
      end
    end
    h[i] = 0
  end

  y = zeros(nlp.meta.ncon)
  for k = 1 : nlp.meta.ncon
    H_errs[k] = Dict{Tuple{Int,Int}, Float64}()
    y[k] = sgn
    Hk = hess(nlp, x, obj_weight=0.0, y=y)
    for i = 1 : nlp.meta.nvar
      h[i] = step
      dJdxi = (jac(nlp, x + h) - jac(nlp, x - h)) / 2 / step
      for j = 1 : i
        err = abs(dJdxi[k, j] - Hk[i, j])
        if err > atol + rtol * abs(dJdxi[k, j])
          H_errs[k][(i,j)] = err
        end
      end
      h[i] = 0
    end
    y[k] = 0
  end

  return H_errs
end
