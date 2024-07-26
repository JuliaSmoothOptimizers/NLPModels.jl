export AbstractNLPModelMeta, NLPModelMeta, reset_data!

"""
    AbstractNLPModelMeta

Base type for metadata related to an optimization model.
"""
abstract type AbstractNLPModelMeta{T, S} end

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
Alternatively, create an `NLPModelMeta` copy from another `AbstractNLPModelMeta` with the ability to change the fields defined in the keyword arguments.
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
struct NLPModelMeta{T, S} <: AbstractNLPModelMeta{T, S}
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

  nlvb::Int
  nlvo::Int
  nlvc::Int

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

  nnzo::Int
  nnzj::Int
  lin_nnzj::Int
  nln_nnzj::Int
  nnzh::Int

  nlin::Int
  nnln::Int

  lin::Vector{Int}
  nln::Vector{Int}

  minimize::Bool
  islp::Bool
  name::String
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
  nln_nnzj = nvar * ncon,
  nnzh = nvar * (nvar + 1) / 2,
  lin = Int[],
  minimize = true,
  islp = false,
  name = "Generic",
) where {T, S}
  if (nvar < 1) || (ncon < 0)
    error("Nonsensical dimensions")
  end

  @lencheck nvar x0 lvar uvar
  @lencheck ncon y0 lcon ucon
  @rangecheck 1 ncon lin
  @assert nnzj == lin_nnzj + nln_nnzj

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

  nnzj = max(0, nnzj)
  nnzh = max(0, nnzh)

  nln = setdiff(1:ncon, lin)
  nlin = length(lin)
  nnln = length(nln)

  NLPModelMeta{T, S}(
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
  )
end

NLPModelMeta(nvar::Int; x0::S = zeros(nvar), kwargs...) where {S} =
  NLPModelMeta{eltype(S), S}(nvar, x0 = x0; kwargs...)

function NLPModelMeta(
  meta::AbstractNLPModelMeta{T, S};
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
  minimize = meta.minimize,
  islp = meta.islp,
  name = meta.name,
) where {T, S}
  NLPModelMeta{T, S}(
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
