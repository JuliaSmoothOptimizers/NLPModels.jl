export MGH01Feas, mgh01feas_autodiff

function mgh01feas_autodiff()

  x0 = [-1.2; 1.0]
  f(x) = zero(eltype(x))
  c(x) = [1 - x[1]; 10 * (x[2] - x[1]^2)]
  lcon = zeros(2)
  ucon = zeros(2)

  return ADNLPModel(f, x0, c, lcon, ucon, name="mgh01feas_autodiff")
end

"""
    nlp = MGH01Feas()

## Rosenbrock function in feasibility format

    Source: Problem 1 in
    J.J. Moré, B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981

```math
\\begin{aligned}
\\min \\quad & 0 \\\\
\\text{s. to} \\quad & 1 - x_1 = 0 \\\\
& 10 (x_2 - x_1^2) = 0.
\\end{aligned}
```

Starting point: `[-1.2; 1]`.
"""
mutable struct MGH01Feas <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
end

function MGH01Feas()
  meta = NLPModelMeta(2, x0=[-1.2; 1.0], name="MGH01Feas_manual", ncon=2, lcon=zeros(2), ucon=zeros(2), nnzj=3, nnzh=1)

  return MGH01Feas(meta, Counters())
end

function NLPModels.obj(nlp :: MGH01Feas, x :: AbstractVector)
  @lencheck 2 x
  increment!(nlp, :neval_obj)
  return zero(eltype(x))
end

function NLPModels.grad!(nlp :: MGH01Feas, x :: AbstractVector{T}, gx :: AbstractVector{T}) where T
  @lencheck 2 x gx
  increment!(nlp, :neval_grad)
  gx .= 0
  return gx
end

function NLPModels.cons!(nls :: MGH01Feas, x :: AbstractVector, cx :: AbstractVector)
  @lencheck 2 x cx
  increment!(nls, :neval_cons)
  cx .= [1 - x[1]; 10 * (x[2] - x[1]^2)]
  return cx
end

# Jx = [-1  0; -20x₁  10]
function NLPModels.jac_structure!(nls :: MGH01Feas, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck 3 rows cols
  rows .= [1, 2, 2]
  cols .= [1, 1, 2]
  return rows, cols
end

function NLPModels.jac_coord!(nls :: MGH01Feas, x :: AbstractVector, vals :: AbstractVector)
  @lencheck 2 x
  @lencheck 3 vals
  increment!(nls, :neval_jac)
  vals .= [-1, -20x[1], 10]
  return vals
end

function NLPModels.jprod!(nls :: MGH01Feas, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck 2 x v Jv
  increment!(nls, :neval_jprod)
  Jv .= [-v[1]; - 20 * x[1] * v[1] + 10 * v[2]]
  return Jv
end

function NLPModels.jtprod!(nls :: MGH01Feas, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck 2 x v Jtv
  increment!(nls, :neval_jtprod)
  Jtv .= [-v[1] - 20 * x[1] * v[2]; 10 * v[2]]
  return Jtv
end

function NLPModels.hess_structure!(nls :: MGH01Feas, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck 1 rows cols
  rows[1] = 1
  cols[1] = 1
  return rows, cols
end

function NLPModels.hess_coord!(nls :: MGH01Feas, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector; obj_weight::Real=one(eltype(x)))
  @lencheck 2 x y
  @lencheck 1 vals
  increment!(nls, :neval_hess)
  vals[1] = -20y[2]
  return vals
end

function NLPModels.hprod!(nls :: MGH01Feas, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight::Real=one(eltype(x)))
  @lencheck 2 x y v Hv
  increment!(nls, :neval_hprod)
  Hv .= [-20y[2] * v[1]; 0]
  return Hv
end

function NLPModels.ghjvprod!(nls :: MGH01Feas, x :: AbstractVector{T}, g :: AbstractVector{T}, v :: AbstractVector{T}, gHv :: AbstractVector{T}) where T
  @lencheck nls.meta.nvar x g v
  @lencheck nls.meta.ncon gHv
  increment!(nls, :neval_hprod)
  gHv .= [T(0); - g[1] * 20v[1]]
  return gHv
end
