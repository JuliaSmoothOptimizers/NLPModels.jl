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

    optimize    objᵢ(x)
    subject to  lvarᵢ ≤    x     ≤ uvarᵢ
                lconᵢ ≤ consᵢ(x) ≤ uconᵢ

for i = 1, ..., nbatch.

Each model variable vector `x` has dimension `nvar`, and constraint vector
`consᵢ(x)` has dimension `ncon`.

All batch data are stored in matrices of size:

- `(nvar, nbatch)` for variables and bounds (`x0`, `lvar`, `uvar`)
- `(ncon, nbatch)` for constraints and multipliers (`y0`, `lcon`, `ucon`)

---

    BatchNLPModelMeta(nbatch::Int, nvar::Int; kwargs...)

Create a `BatchNLPModelMeta` with `nbatch` models, each having `nvar` variables.
The following keyword arguments are accepted:
- `x0`: initial guess
- `lvar`: matrix of lower bounds
- `uvar`: matrix of upper bounds
- `ncon`: number of general constraints
- `y0`: initial Lagrange multipliers
- `lcon`: matrix of constraint lower bounds
- `ucon`: matrix of constraint upper bounds
- `nnzj`: number of elements needed to store the nonzeros in the sparse Jacobian
- `nnzh`: number of elements needed to store the nonzeros in the sparse Hessian
- `minimize`: true if optimize == minimize
- `islp`: true if the problems are linear programs
- `name`: problem name for the batch
- `sparse_jacobian`: indicates whether the Jacobian of the constraints is sparse
- `sparse_hessian`: indicates whether the Hessian of the Lagrangian is sparse
- `grad_available`: indicates whether the gradient of the objective is available
- `jac_available`: indicates whether the Jacobian of the constraints is available
- `hess_available`: indicates whether the Hessian of the Lagrangian is available
- `jprod_available`: indicates whether the Jacobian-vector product `J * v` is available
- `jtprod_available`: indicates whether the transpose Jacobian-vector product `J' * v` is available
- `hprod_available`: indicates whether the Hessian-vector product of the Lagrangian `H * v` is available
"""
struct BatchNLPModelMeta{T, S} <: AbstractBatchNLPModelMeta{T, S}
  nbatch::Int
  nvar::Int
  x0::S
  lvar::S
  uvar::S
  ncon::Int
  y0::S
  lcon::S
  ucon::S
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

for field in fieldnames(BatchNLPModelMeta)
  meth = Symbol("get_", field)
  @eval begin
    $meth(meta::AbstractBatchNLPModelMeta) = getproperty(meta, $(QuoteNode(field)))
  end
  @eval $meth(bnlp::AbstractBatchNLPModel) = $meth(bnlp.meta)
  @eval export $meth
end

function BatchNLPModelMeta{T, S}(
  nbatch::Int,
  nvar::Int;
  x0::S = fill!(S(undef, nvar, nbatch), zero(T)),
  lvar::S = fill!(S(undef, nvar, nbatch), T(-Inf)),
  uvar::S = fill!(S(undef, nvar, nbatch), T(Inf)),
  ncon::Int = 0,
  y0::S = fill!(S(undef, ncon, nbatch), zero(T)),
  lcon::S = fill!(S(undef, ncon, nbatch), T(-Inf)),
  ucon::S = fill!(S(undef, ncon, nbatch), T(Inf)),
  nnzj::Int = nvar * ncon,
  nnzh::Int = nvar * (nvar + 1) ÷ 2,
  minimize::Bool = true,
  islp::Bool = false,
  name::String = "Batch NLP",
  sparse_jacobian::Bool = true,
  sparse_hessian::Bool = true,
  grad_available::Bool = true,
  jac_available::Bool = (ncon > 0),
  hess_available::Bool = true,
  jprod_available::Bool = (ncon > 0),
  jtprod_available::Bool = (ncon > 0),
  hprod_available::Bool = true,
) where {T, S}
  if (nbatch < 1) || (nvar < 1) || (ncon < 0) || (nnzj < 0) || (nnzh < 0)
    error("Nonsensical dimensions")
  end

  BatchNLPModelMeta{T, S}(
    nbatch,
    nvar,
    x0,
    lvar,
    uvar,
    ncon,
    y0,
    lcon,
    ucon,
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
