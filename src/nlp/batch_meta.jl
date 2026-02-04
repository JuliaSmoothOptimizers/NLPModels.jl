export AbstractBatchNLPModelMeta, BatchNLPModelMeta

"""
    AbstractBatchNLPModelMeta

Abstract base type for metadata related to batched nonlinear optimization models.
"""
abstract type AbstractBatchNLPModelMeta{T, S, VI} end

"""
    BatchNLPModelMeta <: AbstractBatchNLPModelMeta

A composite type that represents the main features of a batch of
nonlinear optimization problems sharing the same structure.

Each batch contains `nbatch` independent NLP models of the form:

    optimize    objᵢ(xᵢ)
    subject to  lvarᵢ ≤    xᵢ    ≤ uvarᵢ
                lconᵢ ≤ consᵢ(xᵢ) ≤ uconᵢ

for i = 1, ..., nbatch.

Each model variable vector `xᵢ` has dimension `nvar`, and constraint vector
`consᵢ(xᵢ)` has dimension `ncon`.

All batch data are stored in concatenated vectors of length:

- `nvar * nbatch` for variables and bounds (`x0`, `lvar`, `uvar`)
- `ncon * nbatch` for constraints and multipliers (`y0`, `lcon`, `ucon`)

---

    BatchNLPModelMeta(nbatch::Int, nvar; kwargs...)

Create a `BatchNLPModelMeta` with `nbatch` models, each having `nvar` variables.
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

  ifix = convert(VI, findall(lvar .== uvar))
  ilow = convert(VI, findall((lvar .> T(-Inf)) .& (uvar .== T(Inf))))
  iupp = convert(VI, findall((lvar .== T(-Inf)) .& (uvar .< T(Inf))))
  irng = convert(VI, findall((lvar .> T(-Inf)) .& (uvar .< T(Inf)) .& (lvar .< uvar)))
  ifree = convert(VI, findall((lvar .== T(-Inf)) .& (uvar .== T(Inf))))
  iinf = convert(VI, findall(lvar .> uvar))

  if ncon > 0
    jfix = convert(VI, findall(lcon .== ucon))
    jlow = convert(VI, findall((lcon .> T(-Inf)) .& (ucon .== T(Inf))))
    jupp = convert(VI, findall((lcon .== T(-Inf)) .& (ucon .< T(Inf))))
    jrng = convert(VI, findall((lcon .> T(-Inf)) .& (ucon .< T(Inf)) .& (lcon .< ucon)))
    jfree = convert(VI, findall((lcon .== T(-Inf)) .& (ucon .== T(Inf))))
    jinf = convert(VI, findall(lcon .> ucon))
  else
    jfix = VI(undef, 0)
    jlow = VI(undef, 0)
    jupp = VI(undef, 0)
    jrng = VI(undef, 0)
    jfree = VI(undef, 0)
    jinf = VI(undef, 0)
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
