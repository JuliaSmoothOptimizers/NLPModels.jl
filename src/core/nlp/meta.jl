export AbstractNLPModelMeta, NLPModelMeta, reset_data!

# Base type for metadata related to an optimization model.
abstract type AbstractNLPModelMeta end

struct NLPModelMeta <: AbstractNLPModelMeta

  # A composite type that represents the main features of
  # the optimization problem
  #
  #  optimize   obj(x)
  #  subject to lvar ≤    x    ≤ uvar
  #             lcon ≤ cons(x) ≤ ucon
  #
  # where x        is an nvar-dimensional vector,
  #       obj      is the real-valued objective function,
  #       cons     is the vector-valued constraint function,
  #       optimize is either "minimize" or "maximize".
  #
  # Here, lvar, uvar, lcon and ucon are vectors. Some of their
  # components may be infinite to indicate that the corresponding
  # bound or general constraint is not present.

  nvar :: Int               # number of variables
  x0   :: Vector    # initial guess
  lvar :: Vector    # vector of lower bounds
  uvar :: Vector    # vector of upper bounds

  ifix  :: Vector{Int}     # indices of fixed variables
  ilow  :: Vector{Int}     # indices of variables with lower bound only
  iupp  :: Vector{Int}     # indices of variables with upper bound only
  irng  :: Vector{Int}     # indices of variables with lower and upper bound (range)
  ifree :: Vector{Int}     # indices of free variables
  iinf  :: Vector{Int}     # indices of infeasible bounds

  nbv   :: Int              # number of linear binary variables
  niv   :: Int              # number of linear non-binary integer variables
  nlvb  :: Int              # number of nonlinear variables in both objectives and constraints
  nlvo  :: Int              # number of nonlinear variables in objectives (includes nlvb)
  nlvc  :: Int              # number of nonlinear variables in constraints (includes nlvb)
  nlvbi :: Int              # number of integer nonlinear variables in both objectives and constraints
  nlvci :: Int              # number of integer nonlinear variables in constraints only
  nlvoi :: Int              # number of integer nonlinear variables in objectives only
  nwv   :: Int              # number of linear network (arc) variables

  ncon :: Int               # number of general constraints
  y0   :: Vector    # initial Lagrange multipliers
  lcon :: Vector    # vector of constraint lower bounds
  ucon :: Vector    # vector of constraint upper bounds

  jfix  :: Vector{Int}     # indices of equality constraints
  jlow  :: Vector{Int}     # indices of constraints of the form c(x) ≥ cl
  jupp  :: Vector{Int}     # indices of constraints of the form c(x) ≤ cu
  jrng  :: Vector{Int}     # indices of constraints of the form cl ≤ c(x) ≤ cu
  jfree :: Vector{Int}     # indices of "free" constraints (there shouldn't be any)
  jinf  :: Vector{Int}     # indices of the visibly infeasible constraints

  nnzo :: Int               # number of nonzeros in all objectives gradients
  nnzj :: Int               # number of elements needed to store the nonzeros in the sparse Jacobian
  nnzh :: Int               # number of elements needed to store the nonzeros in the sparse Hessian

  nlin  :: Int              # number of linear constraints
  nnln  :: Int              # number of nonlinear general constraints
  nnnet :: Int              # number of nonlinear network constraints
  nlnet :: Int              # number of linear network constraints

  lin   :: Vector{Int}     # indices of linear constraints
  nln   :: Vector{Int}     # indices of nonlinear constraints
  nnet  :: Vector{Int}     # indices of nonlinear network constraints
  lnet  :: Vector{Int}     # indices of linear network constraints

  minimize :: Bool          # true if optimize == minimize
  nlo  :: Int               # number of nonlinear objectives
  islp :: Bool              # true if the problem is a linear program
  name :: String       # problem name

  function NLPModelMeta(nvar;
                        x0=zeros(nvar,),
                        lvar=-Inf * ones(nvar,),
                        uvar=Inf * ones(nvar,),
                        nbv=0,
                        niv=0,
                        nlvb=nvar,
                        nlvo=nvar,
                        nlvc=nvar,
                        nlvbi=0,
                        nlvci=0,
                        nlvoi=0,
                        nwv=0,
                        ncon=0,
                        y0=zeros(ncon,),
                        lcon=-Inf * ones(ncon,),
                        ucon=Inf * ones(ncon,),
                        nnzo=nvar,
                        nnzj=nvar * ncon,
                        nnzh=nvar * (nvar + 1) / 2,
                        lin=Int[],
                        nln=1:ncon,
                        nnet=Int[],
                        lnet=Int[],
                        nlin=length(lin),
                        nnln=length(nln),
                        nnnet=length(nnet),
                        nlnet=length(lnet),
                        minimize=true,
                        nlo=1,
                        islp=false,
                        name="Generic")
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

    ifix  = findall(lvar .== uvar)
    ilow  = findall((lvar .> -Inf) .& (uvar .== Inf))
    iupp  = findall((lvar .== -Inf) .& (uvar .< Inf))
    irng  = findall((lvar .> -Inf) .& (uvar .< Inf) .& (lvar .< uvar))
    ifree = findall((lvar .== -Inf) .& (uvar .== Inf))
    iinf  = findall(lvar .> uvar)

    jfix  = findall(lcon .== ucon)
    jlow  = findall((lcon .> -Inf) .& (ucon .== Inf))
    jupp  = findall((lcon .== -Inf) .& (ucon .< Inf))
    jrng  = findall((lcon .> -Inf) .& (ucon .< Inf) .& (lcon .< ucon))
    jfree = findall((lcon .== -Inf) .& (ucon .== Inf))
    jinf  = findall(lcon .> ucon)

    nnzj = max(0, nnzj)
    nnzh = max(0, nnzh)

    new(nvar, x0, lvar, uvar,
        ifix, ilow, iupp, irng, ifree, iinf,
        nbv, niv, nlvb, nlvo, nlvc,
        nlvbi, nlvci, nlvoi, nwv,
        ncon, y0, lcon, ucon,
        jfix, jlow, jupp, jrng, jfree, jinf,
        nnzo, nnzj, nnzh,
        nlin, nnln, nnnet, nlnet, lin, nln, nnet, lnet,
        minimize, nlo, islp, name)
  end
end

"""
    reset_data!(nlp)

Reset model data if appropriate.
This method should be overloaded if a subtype of `AbstractNLPModel`
contains data that should be reset, such as a quasi-Newton linear
operator.
"""
function reset_data!(nlp :: AbstractNLPModel)
  return nlp
end