export AbstractNLPModelMeta, NLPModelMeta, reset_data!

"""
    AbstractNLPModelMeta

Base type for metadata related to an optimization model.
"""
abstract type AbstractNLPModelMeta{T, S, VI} end

"""
    NLPModelMeta <: AbstractNLPModelMeta

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

    NLPModelMeta(nvar::Integer; kwargs...)
    NLPModelMeta(meta::AbstractNLPModelMeta; kwargs...)

Create an `NLPModelMeta` with `nvar` variables.
Alternatively, create an `NLPModelMeta` copy from another `AbstractNLPModelMeta`.
The following keyword arguments are accepted:
- `x0`: initial guess
- `lvar`: vector of lower bounds
- `uvar`: vector of upper bounds
- `nlvb`: number of nonlinear variables in both objectives and constraints
- `nlvo`: number of nonlinear variables in objectives (includes nlvb)
- `nlvc`: number of nonlinear variables in constraints (includes nlvb)
- `ncon`: number of general constraints
- `y0`: initial Lagrange multipliers
- `lcon`: vector of constraint lower bounds
- `ucon`: vector of constraint upper bounds
- `nnzo`: number of nonzeros in the gradient
- `nnzj`: number of elements needed to store the nonzeros in the sparse Jacobian
- `lin_nnzj`: number of elements needed to store the nonzeros in the sparse Jacobian of linear constraints
- `nln_nnzj`: number of elements needed to store the nonzeros in the sparse Jacobian of nonlinear constraints
- `nnzh`: number of elements needed to store the nonzeros in the sparse Hessian
- `lin`: indices of linear constraints
- `minimize`: true if optimize == minimize
- `islp`: true if the problem is a linear program
- `name`: problem name
- `grad_available`: indicates whether the gradient of the objective is available
- `jac_available`: indicates whether the sparse Jacobian of the constraints is available
- `hess_available`: indicates whether the sparse Hessian of the Lagrangian is available
- `jprod_available`: indicates whether the Jacobian-vector product `J * v` is available
- `jtprod_available`: indicates whether the transpose Jacobian-vector product `J' * v` is available
- `hprod_available`: indicates whether the Hessian-vector product of the Lagrangian `H * v` is available

`NLPModelMeta` also contains the following attributes, which are computed from the variables above:
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
- `nlin`: number of linear constraints
- `nnln`: number of nonlinear general constraints
- `nln`: indices of nonlinear constraints
"""
struct NLPModelMeta{T, S, VI} <: AbstractNLPModelMeta{T, S, VI}
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

  nlvb::Int
  nlvo::Int
  nlvc::Int

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

  nnzo::Int
  nnzj::Int
  lin_nnzj::Int
  nln_nnzj::Int
  nnzh::Int

  nlin::Int
  nnln::Int

  lin::VI
  nln::VI

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

function NLPModelMeta{T, S}(
  nvar::Int;
  x0::S = fill!(S(undef, nvar), zero(T)),
  lvar::S = fill!(S(undef, nvar), T(-Inf)),
  uvar::S = fill!(S(undef, nvar), T(Inf)),
  nlvb = nvar,
  nlvo = nvar,
  nlvc = nvar,
  ncon = 0,
  y0::S = fill!(S(undef, ncon), zero(T)),
  lcon::S = fill!(S(undef, ncon), T(-Inf)),
  ucon::S = fill!(S(undef, ncon), T(Inf)),
  nnzo = nvar,
  nnzj = nvar * ncon,
  lin_nnzj = 0,
  nln_nnzj = nnzj - lin_nnzj,
  nnzh = nvar * (nvar + 1) ÷ 2,
  lin = Int[],
  minimize::Bool = true,
  islp::Bool = false,
  name = "Generic",
  grad_available::Bool = true,
  jac_available::Bool = (ncon > 0),
  hess_available::Bool = true,
  jprod_available::Bool = (ncon > 0),
  jtprod_available::Bool = (ncon > 0),
  hprod_available::Bool = true,
) where {T, S}
  NLPModelMeta{T, S, Vector{Int}}(
    nvar;
    x0,
    lvar,
    uvar,
    nlvb,
    nlvo,
    nlvc,
    ncon,
    y0,
    lcon,
    ucon,
    nnzo,
    nnzj,
    lin_nnzj,
    nln_nnzj,
    nnzh,
    lin,
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

function NLPModelMeta{T, S, VI}(
  nvar::Int;
  x0::S = fill!(S(undef, nvar), zero(T)),
  lvar::S = fill!(S(undef, nvar), T(-Inf)),
  uvar::S = fill!(S(undef, nvar), T(Inf)),
  nlvb = nvar,
  nlvo = nvar,
  nlvc = nvar,
  ncon = 0,
  y0::S = fill!(S(undef, ncon), zero(T)),
  lcon::S = fill!(S(undef, ncon), T(-Inf)),
  ucon::S = fill!(S(undef, ncon), T(Inf)),
  nnzo = nvar,
  nnzj = nvar * ncon,
  lin_nnzj = 0,
  nln_nnzj = nnzj - lin_nnzj,
  nnzh = nvar * (nvar + 1) ÷ 2,
  lin = VI(undef, 0),
  minimize::Bool = true,
  islp::Bool = false,
  name = "Generic",
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

  @lencheck nvar x0 lvar uvar
  @lencheck ncon y0 lcon ucon
  @rangecheck 1 ncon lin
  @assert nnzj == lin_nnzj + nln_nnzj

  ifix = VI(undef, nvar)
  ilow = VI(undef, nvar)
  iupp = VI(undef, nvar)
  irng = VI(undef, nvar)
  ifree = VI(undef, nvar)
  iinf = VI(undef, nvar)

  map!((lv, uv) -> lv == uv, ifix, lvar, uvar)
  map!((lv, uv) -> (lv > T(-Inf)) & (uv == T(Inf)), ilow, lvar, uvar)
  map!((lv, uv) -> (lv == T(-Inf)) & (uv < T(Inf)), iupp, lvar, uvar)
  map!((lv, uv) -> (lv > T(-Inf)) & (uv < T(Inf)), irng, lvar, uvar)
  map!((lv, uv) -> (lv == T(-Inf)) & (uv == T(Inf)), ifree, lvar, uvar)
  map!((lv, uv) -> lv > uv, iinf, lvar, uvar)

  jfix = VI(undef, ncon)
  jlow = VI(undef, ncon)
  jupp = VI(undef, ncon)
  jrng = VI(undef, ncon)
  jfree = VI(undef, ncon)
  jinf = VI(undef, ncon)

  if ncon > 0
    map!((lc, uc) -> lc == uc, jfix, lcon, ucon)
    map!((lc, uc) -> (lc > T(-Inf)) & (uc == T(Inf)), jlow, lcon, ucon)
    map!((lc, uc) -> (lc == T(-Inf)) & (uc < T(Inf)), jupp, lcon, ucon)
    map!((lc, uc) -> (lc > T(-Inf)) & (uc < T(Inf)), jrng, lcon, ucon)
    map!((lc, uc) -> (lc == T(-Inf)) & (uc == T(Inf)), jfree, lcon, ucon)
    map!((lc, uc) -> lc > uc, jinf, lcon, ucon)
  end

  nln = setdiff(VI(1:ncon), lin)
  nlin = length(lin)
  nnln = length(nln)

  NLPModelMeta{T, S, VI}(
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
    nlvb,
    nlvo,
    nlvc,
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
    nnzo,
    nnzj,
    lin_nnzj,
    nln_nnzj,
    nnzh,
    nlin,
    nnln,
    lin,
    nln,
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

NLPModelMeta(nvar::Int; x0::S = zeros(nvar), kwargs...) where {S} =
  NLPModelMeta{eltype(S), S, Vector{Int}}(nvar, x0 = x0; kwargs...)

function NLPModelMeta(
  meta::AbstractNLPModelMeta{T, S, VI};
  nvar::Int = meta.nvar,
  x0::S = meta.x0,
  lvar::S = meta.lvar,
  uvar::S = meta.uvar,
  nlvb = meta.nlvb,
  nlvo = meta.nlvo,
  nlvc = meta.nlvc,
  ncon = meta.ncon,
  y0::S = meta.y0,
  lcon::S = meta.lcon,
  ucon::S = meta.ucon,
  nnzo = meta.nnzo,
  nnzj = meta.nnzj,
  lin_nnzj = meta.lin_nnzj,
  nln_nnzj = meta.nln_nnzj,
  nnzh = meta.nnzh,
  lin = meta.lin,
  minimize::Bool = meta.minimize,
  islp::Bool = meta.islp,
  name = meta.name,
  grad_available::Bool = meta.grad_available,
  jac_available::Bool = meta.jac_available,
  hess_available::Bool = meta.hess_available,
  jprod_available::Bool = meta.jprod_available,
  jtprod_available::Bool = meta.jtprod_available,
  hprod_available::Bool = meta.hprod_available,
) where {T, S, VI}
  NLPModelMeta{T, S, VI}(
    nvar,
    x0 = x0,
    lvar = lvar,
    uvar = uvar,
    nlvb = nlvb,
    nlvo = nlvo,
    nlvc = nlvc,
    ncon = ncon,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
    nnzo = nnzo,
    nnzj = nnzj,
    lin_nnzj = lin_nnzj,
    nln_nnzj = nln_nnzj,
    nnzh = nnzh,
    lin = lin,
    minimize = minimize,
    islp = islp,
    name = name,
    grad_available = grad_available,
    jac_available = jac_available,
    hess_available = hess_available,
    jprod_available = jprod_available,
    jtprod_available = jtprod_available,
    hprod_available = hprod_available,
  )
end

"""
    reset_data!(nlp)

Reset model data if appropriate.
This method should be overloaded if a subtype of `AbstractNLPModel`
contains data that should be reset, such as a quasi-Newton linear
operator.
"""
function reset_data!(nlp::AbstractNLPModel)
  return nlp
end
