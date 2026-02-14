export AbstractBatchNLPModelMeta, BatchNLPModelMeta

"""
    AbstractBatchNLPModelMeta

Abstract base type for metadata related to batched nonlinear optimization models.
"""
abstract type AbstractBatchNLPModelMeta{T, S} end

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

    BatchNLPModelMeta(nbatch::Int, nvar::Int; kwargs...)

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
- `sparse_jacobian`: indicates whether the Jacobian of the constraints is sparse
- `sparse_hessian`: indicates whether the Hessian of the Lagrangian is sparse
- `grad_available`: indicates whether the gradient of the objective is available
- `jac_available`: indicates whether the Jacobian of the constraints is available
- `hess_available`: indicates whether the Hessian of the Lagrangian is available
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
struct BatchNLPModelMeta{T, S} <: AbstractBatchNLPModelMeta{T, S}
  nbatch::Int
  nvar::Int
  x0::S
  lvar::S
  uvar::S

  ifix::Vector{Int}
  ilow::Vector{Int}
  iupp::Vector{Int}
  irng::Vector{Int}
  ifree::Vector{Int}
  iinf::Vector{Int}

  ncon::Int
  y0::S
  lcon::S
  ucon::S

  jfix::Vector{Int}
  jlow::Vector{Int}
  jupp::Vector{Int}
  jrng::Vector{Int}
  jfree::Vector{Int}
  jinf::Vector{Int}

  nnzj::Int
  nnzh::Int

  minimize::Bool
  islp::Bool
  name::String

  sparse_jacobian::Bool
  sparse_hessian::Bool

  grad_available::Bool
  jac_available::Bool
  hess_available::Bool
  jprod_available::Bool
  jtprod_available::Bool
  hprod_available::Bool
end

function BatchNLPModelMeta{T, S}(
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
  sparse_jacobian::Bool,
  sparse_hessian::Bool,
  grad_available::Bool = true,
  jac_available::Bool = (ncon > 0),
  hess_available::Bool = true,
  jprod_available::Bool = (ncon > 0),
  jtprod_available::Bool = (ncon > 0),
  hprod_available::Bool = true,
) where {T, S}
  if (nvar < 1) || (ncon < 0) || (nnzj < 0) || (nnzh < 0)
    error("Nonsensical dimensions")
  end

  ifix = findall(lvar .== uvar)
  ilow = findall((lvar .> T(-Inf)) .& (uvar .== T(Inf)))
  iupp = findall((lvar .== T(-Inf)) .& (uvar .< T(Inf)))
  irng = findall((lvar .> T(-Inf)) .& (uvar .< T(Inf)) .& (lvar .< uvar))
  ifree = findall((lvar .== T(-Inf)) .& (uvar .== T(Inf)))
  iinf = findall(lvar .> uvar)

  if ncon > 0
    jfix = findall(lcon .== ucon)
    jlow = findall((lcon .> T(-Inf)) .& (ucon .== T(Inf)))
    jupp = findall((lcon .== T(-Inf)) .& (ucon .< T(Inf)))
    jrng = findall((lcon .> T(-Inf)) .& (ucon .< T(Inf)) .& (lcon .< ucon))
    jfree = findall((lcon .== T(-Inf)) .& (ucon .== T(Inf)))
    jinf = findall(lcon .> ucon)
  else
    jfix = Int[]
    jlow = Int[]
    jupp = Int[]
    jrng = Int[]
    jfree = Int[]
    jinf = Int[]
  end

  BatchNLPModelMeta{T, S}(
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
    sparse_jacobian,
    sparse_hessian,
    grad_available,
    jac_available,
    hess_available,
    jprod_available,
    jtprod_available,
    hprod_available,
  )
end
