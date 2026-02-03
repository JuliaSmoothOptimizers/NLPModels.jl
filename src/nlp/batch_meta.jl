export AbstractBatchNLPModelMeta, BatchNLPModelMeta

"""
    AbstractBatchNLPModelMeta

Abstract base type for metadata related to batched nonlinear optimization models.
"""
abstract type AbstractBatchNLPModelMeta{T, S, VI} end

"""
    BatchNLPModelMeta <: AbstractBatchNLPModelMeta

A composite type that represents the main features of the optimization problem

    optimize    obj(x)
    subject to  lvar ≤    x    ≤ uvar
                lcon ≤ cons(x) ≤ ucon

where `x`        is an `nvar`-dimensional vector,
      `obj`      is the real-valued objective function,
      `cons`     is the vector-valued constraint function,
      `optimize` is either "minimize" or "maximize".

Here, `lvar`, `uvar`, `lcon` and `ucon` are vectors.
Some of their components may be infinite to indicate that the corresponding bound or general constraint is not present.

---

    BatchNLPModelMeta(nbatch::Int, nvar; kwargs...)

Create an `BatchNLPModelMeta` with `nvar` variables.
Alternatively, create an `NLPModelMeta` copy from another `AbstractNLPModelMeta`.
The following keyword arguments are accepted:
- `x0`: initial guess
- `lvar`: vector of lower bounds
- `uvar`: vector of upper bounds
- `ncon`: number of general constraints
- `y0`: initial Lagrange multipliers
- `lcon`: vector of constraint lower bounds
- `ucon`: vector of constraint upper bounds
- `nnzj`: number of elements needed to store the nonzeros in the sparse Jacobian
- `nnzh`: number of elements needed to store the nonzeros in the sparse Hessian
- `minimize`: true if optimize == minimize
- `islp`: true if the problem is a linear program
- `name`: problem name
- `grad_available`: indicates whether the gradient of the objective is available
- `jac_available`: indicates whether the sparse Jacobian of the constraints is available
- `hess_available`: indicates whether the sparse Hessian of the Lagrangian is available
- `jprod_available`: indicates whether the Jacobian-vector product `J * v` is available
- `jtprod_available`: indicates whether the transpose Jacobian-vector product `J' * v` is available
- `hprod_available`: indicates whether the Hessian-vector product of the Lagrangian `H * v` is available

`BatchNLPModelMeta` also contains the following attributes, which are computed from the variables above:
- `nbatch`: number of models
- `nvar`: number of variables
- `ifix`: indices of fixed variables
- `ilow`: indices of variables with lower bound only
- `iupp`: indices of variables with upper bound only
- `irng`: indices of variables with lower and upper bound (range)
- `ifree`: indices of free variables
- `iinf`: indices of visibly infeasible bounds
- `jfix`: indices of equality constraints
- `jlow`: indices of constraints of the form c(x) ≥ cl
- `jupp`: indices of constraints of the form c(x) ≤ cu
- `jrng`: indices of constraints of the form cl ≤ c(x) ≤ cu
- `jfree`: indices of "free" constraints (there shouldn't be any)
- `jinf`: indices of the visibly infeasible constraints
"""
struct BatchNLPModelMeta{T, S, VI} <: AbstractBatchNLPModelMeta{T, S, VI}
  nbatch::Int
  nvar::Int
  x0::S
  lvar::S
  uvar::S

  ifix::VI
  ilow::VI
  iupp::VI
  irng::VI
  ifree::VI
  iinf::VI

  ncon::Int
  y0::S
  lcon::S
  ucon::S

  jfix::VI
  jlow::VI
  jupp::VI
  jrng::VI
  jfree::VI
  jinf::VI

  nnzj::Int
  nnzh::Int

  minimize::Bool
  islp::Bool
  name::String

  grad_available::Bool
  jac_available::Bool
  hess_available::Bool
  jprod_available::Bool
  jtprod_available::Bool
  hprod_available::Bool
end

function BatchNLPModelMeta{T, S, VI}(
  nbatch::Int,
  nvar::Int;
  x0::S = fill!(S(undef, nvar * nbatch), zero(T)),
  lvar::S = fill!(S(undef, nvar * nbatch), T(-Inf)),
  uvar::S = fill!(S(undef, nvar * nbatch), T(Inf)),
  ncon::Int = 0,
  y0::S = fill!(S(undef, ncon * nbatch), zero(T)),
  lcon::S = fill!(S(undef, ncon * nbatch), T(-Inf)),
  ucon::S = fill!(S(undef, ncon * nbatch), T(Inf)),
  nnzj::Int = nvar * ncon,
  nnzh::Int = nvar * (nvar + 1) ÷ 2,
  minimize::Bool = true,
  islp::Bool = false,
  name::String = "Batch NLP",
  grad_available::Bool = true,
  jac_available::Bool = (ncon > 0),
  hess_available::Bool = true,
  jprod_available::Bool = (ncon > 0),
  jtprod_available::Bool = (ncon > 0),
  hprod_available::Bool = true,
) where {T, S, VI}
  if (nvar < 1) || (ncon < 0) || (nnzj < 0) || (nnzh < 0)
    error("Nonsensical dimensions")
  end

  ifix = VI(undef, nvar * nbatch)
  ilow = VI(undef, nvar * nbatch)
  iupp = VI(undef, nvar * nbatch)
  irng = VI(undef, nvar * nbatch)
  ifree = VI(undef, nvar * nbatch)
  iinf = VI(undef, nvar * nbatch)

  map!((lv, uv) -> lv == uv, ifix, lvar, uvar)
  map!((lv, uv) -> (lv > T(-Inf)) & (uv == T(Inf)), ilow, lvar, uvar)
  map!((lv, uv) -> (lv == T(-Inf)) & (uv < T(Inf)), iupp, lvar, uvar)
  map!((lv, uv) -> (lv > T(-Inf)) & (uv < T(Inf)), irng, lvar, uvar)
  map!((lv, uv) -> (lv == T(-Inf)) & (uv == T(Inf)), ifree, lvar, uvar)
  map!((lv, uv) -> lv > uv, iinf, lvar, uvar)

  jfix = VI(undef, ncon * nbatch)
  jlow = VI(undef, ncon * nbatch)
  jupp = VI(undef, ncon * nbatch)
  jrng = VI(undef, ncon * nbatch)
  jfree = VI(undef, ncon * nbatch)
  jinf = VI(undef, ncon * nbatch)

  if ncon > 0
    map!((lc, uc) -> lc == uc, jfix, lcon, ucon)
    map!((lc, uc) -> (lc > T(-Inf)) & (uc == T(Inf)), jlow, lcon, ucon)
    map!((lc, uc) -> (lc == T(-Inf)) & (uc < T(Inf)), jupp, lcon, ucon)
    map!((lc, uc) -> (lc > T(-Inf)) & (uc < T(Inf)), jrng, lcon, ucon)
    map!((lc, uc) -> (lc == T(-Inf)) & (uc == T(Inf)), jfree, lcon, ucon)
    map!((lc, uc) -> lc > uc, jinf, lcon, ucon)
  end

  BatchNLPModelMeta{T, S, VI}(
    nbatch,
    nvar,
    x0,
    lvar,
    uvar,
    ifix,
    ilow,
    iupp,
    irng,
    ifree,
    iinf,
    ncon,
    y0,
    lcon,
    ucon,
    jfix,
    jlow,
    jupp,
    jrng,
    jfree,
    jinf,
    nnzj,
    nnzh,
    minimize,
    islp,
    name,
    grad_available,
    jac_available,
    hess_available,
    jprod_available,
    jtprod_available,
    hprod_available,
  )
end
