export AbstractNLPModelMeta, NLPModelMeta, reset_data!

"""
    AbstractNLPModelMeta

Base type for metadata related to an optimization model.
"""
abstract type AbstractNLPModelMeta{T,S} end

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

    NLPModelMeta(nvar; kwargs...)

Create an `NLPModelMeta` with `nvar` variables.
The following keyword arguments are accepted:
- `x0`: initial guess
- `lvar`: vector of lower bounds
- `uvar`: vector of upper bounds
- `nbv`: number of linear binary variables
- `niv`: number of linear non-binary integer variables
- `nlvb`: number of nonlinear variables in both objectives and constraints
- `nlvo`: number of nonlinear variables in objectives (includes nlvb)
- `nlvc`: number of nonlinear variables in constraints (includes nlvb)
- `nlvbi`: number of integer nonlinear variables in both objectives and constraints
- `nlvci`: number of integer nonlinear variables in constraints only
- `nlvoi`: number of integer nonlinear variables in objectives only
- `nwv`: number of linear network (arc) variables
- `ncon`: number of general constraints
- `y0`: initial Lagrange multipliers
- `lcon`: vector of constraint lower bounds
- `ucon`: vector of constraint upper bounds
- `nnzo`: number of nonzeros in all objectives gradients
- `nnzj`: number of elements needed to store the nonzeros in the sparse Jacobian
- `nnzh`: number of elements needed to store the nonzeros in the sparse Hessian
- `nlin`: number of linear constraints
- `nnln`: number of nonlinear general constraints
- `nnnet`: number of nonlinear network constraints
- `nlnet`: number of linear network constraints
- `lin`: indices of linear constraints
- `nln`: indices of nonlinear constraints
- `nnet`: indices of nonlinear network constraints
- `lnet`: indices of linear network constraints
- `minimize`: true if optimize == minimize
- `nlo`: number of nonlinear objectives
- `islp`: true if the problem is a linear program
- `name`: problem name
"""
struct NLPModelMeta{T,S} <: AbstractNLPModelMeta{T,S}
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

  nbv::Int
  niv::Int
  nlvb::Int
  nlvo::Int
  nlvc::Int
  nlvbi::Int
  nlvci::Int
  nlvoi::Int
  nwv::Int

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
  nnzh::Int

  nlin::Int
  nnln::Int
  nnnet::Int
  nlnet::Int

  lin::Vector{Int}
  nln::Vector{Int}
  nnet::Vector{Int}
  lnet::Vector{Int}

  minimize::Bool
  nlo::Int
  islp::Bool
  name::String

  function NLPModelMeta{T,S}(
    nvar::Int;
    x0::S = zeros(T, nvar),
    lvar::S = T(-Inf) * ones(T, nvar),
    uvar::S = T(Inf) * ones(T, nvar),
    nbv = 0,
    niv = 0,
    nlvb = nvar,
    nlvo = nvar,
    nlvc = nvar,
    nlvbi = 0,
    nlvci = 0,
    nlvoi = 0,
    nwv = 0,
    ncon = 0,
    y0::S = zeros(T, ncon),
    lcon::S = T(-Inf) * ones(T, ncon),
    ucon::S = T(Inf) * ones(T, ncon),
    nnzo = nvar,
    nnzj = nvar * ncon,
    nnzh = nvar * (nvar + 1) / 2,
    lin = Int[],
    nln = 1:ncon,
    nnet = Int[],
    lnet = Int[],
    nlin = length(lin),
    nnln = length(nln),
    nnnet = length(nnet),
    nlnet = length(lnet),
    minimize = true,
    nlo = 1,
    islp = false,
    name = "Generic",
  ) where {T,S}

    if (nvar < 1) || (ncon < 0)
      error("Nonsensical dimensions")
    end

    @lencheck nvar x0 lvar uvar
    @lencheck ncon y0 lcon ucon
    @lencheck nlin lin
    @lencheck nnln nln
    @lencheck nnnet nnet
    @lencheck nlnet lnet
    @rangecheck 1 ncon lin nln nnet lnet
    # T = eltype(x0)

    ifix = findall(lvar .== uvar)
    ilow = findall((lvar .> T(-Inf)) .& (uvar .== T(Inf)))
    iupp = findall((lvar .== T(-Inf)) .& (uvar .< T(Inf)))
    irng = findall((lvar .> T(-Inf)) .& (uvar .< T(Inf)) .& (lvar .< uvar))
    ifree = findall((lvar .== T(-Inf)) .& (uvar .== T(Inf)))
    iinf = findall(lvar .> uvar)

    jfix = findall(lcon .== ucon)
    jlow = findall((lcon .> T(-Inf)) .& (ucon .== T(Inf)))
    jupp = findall((lcon .== T(-Inf)) .& (ucon .< T(Inf)))
    jrng = findall((lcon .> T(-Inf)) .& (ucon .< T(Inf)) .& (lcon .< ucon))
    jfree = findall((lcon .== T(-Inf)) .& (ucon .== T(Inf)))
    jinf = findall(lcon .> ucon)

    nnzj = max(0, nnzj)
    nnzh = max(0, nnzh)

    new{T,S}(
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
      nbv,
      niv,
      nlvb,
      nlvo,
      nlvc,
      nlvbi,
      nlvci,
      nlvoi,
      nwv,
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
      nnzh,
      nlin,
      nnln,
      nnnet,
      nlnet,
      lin,
      nln,
      nnet,
      lnet,
      minimize,
      nlo,
      islp,
      name,
    )
  end
end

NLPModelMeta(nvar; x0::S = zeros(nvar), kwargs...) where S = NLPModelMeta{eltype(S), S}(nvar, x0 = x0; kwargs...)

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
