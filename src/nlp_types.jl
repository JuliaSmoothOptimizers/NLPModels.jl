export has_bounds, bound_constrained, unconstrained, linearly_constrained,
      equality_constrained, inequality_constrained

# Base type for an optimization model.
abstract type AbstractNLPModel end

mutable struct Counters
  neval_obj    :: Int  # Number of objective evaluations.
  neval_grad   :: Int  # Number of objective gradient evaluations.
  neval_cons   :: Int  # Number of constraint vector evaluations.
  neval_jcon   :: Int  # Number of individual constraint evaluations.
  neval_jgrad  :: Int  # Number of individual constraint gradient evaluations.
  neval_jac    :: Int  # Number of constraint Jacobian evaluations.
  neval_jprod  :: Int  # Number of Jacobian-vector products.
  neval_jtprod :: Int  # Number of transposed Jacobian-vector products.
  neval_hess   :: Int  # Number of Lagrangian/objective Hessian evaluations.
  neval_hprod  :: Int  # Number of Lagrangian/objective Hessian-vector products.
  neval_jhprod :: Int  # Number of individual constraint Hessian-vector products.

  function Counters()
    return new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
  end
end

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

# Displaying NLPModelMeta instances.

import Base.show, Base.print, Base.println
function show(io :: IO, nlp :: NLPModelMeta)
  s  = nlp.minimize ? @sprintf("Minimization ") : @sprintf("Maximization ")
  s *= @sprintf("problem %s\n", nlp.name)
  s *= @sprintf("nvar = %d, ncon = %d (%d linear)\n", nlp.nvar, nlp.ncon, nlp.nlin)
  print(io, s)
end

"""
    print(io, meta)

Prints meta information - x0, nvar, ncon, etc.
"""
function print(io :: IO, nlp :: NLPModelMeta)
  nlp.minimize ? @printf(io, "Minimization ") : @printf(io, "Maximization ")
  dsp(x) = length(x) == 0 ? print(io, "∅") :
    (length(x) <= 5 ? Base.show_delim_array(io, x, "", "  ", "", false) :
     begin
      Base.show_delim_array(io, x[1:4], "", "  ", "", false)
      print("  ⋯  $(x[end])")
    end)
  @printf(io, "problem %s\n", nlp.name)
  @printf(io, "nvar = %d, ncon = %d (%d linear)\n", nlp.nvar, nlp.ncon, nlp.nlin)
  @printf(io, "lvar = "); dsp(nlp.lvar'); @printf(io, "\n")
  @printf(io, "uvar = "); dsp(nlp.uvar'); @printf(io, "\n")
  @printf(io, "lcon = "); dsp(nlp.lcon'); @printf(io, "\n")
  @printf(io, "ucon = "); dsp(nlp.ucon'); @printf(io, "\n")
  @printf(io, "x0 = ");   dsp(nlp.x0'); @printf(io, "\n")
  @printf(io, "y0 = ");   dsp(nlp.y0'); @printf(io, "\n")
  @printf(io, "nnzh = %d\n", nlp.nnzh)
  @printf(io, "nnzj = %d\n", nlp.nnzj)
  if nlp.nlin > 0
    @printf(io, "linear constraints:    "); dsp(nlp.lin'); @printf(io, "\n")
  end
  if nlp.nnln > 0
    @printf(io, "nonlinear constraints: "); dsp(nlp.nln'); @printf(io, "\n")
  end
  if nlp.nlnet > 0
    @printf(io, "linear network constraints:   "); dsp(nlp.lnet'); @printf(io, "\n")
  end
  if nlp.nnnet > 0
    @printf(io, "nonlinear network constraints:   "); dsp(nlp.nnet'); @printf(io, "\n")
  end
end

"""
    has_bounds(nlp)
    has_bounds(meta)

Returns whether the problem has bounds on the variables.
"""
has_bounds(meta::NLPModelMeta) = length(meta.ifree) < meta.nvar

"""
    bound_constrained(nlp)
    bound_constrained(meta)

Returns whether the problem has bounds on the variables and no other constraints.
"""
bound_constrained(meta::NLPModelMeta) = meta.ncon == 0 && has_bounds(meta)

"""
    unconstrained(nlp)
    unconstrained(meta)

Returns whether the problem in unconstrained.
"""
unconstrained(meta::NLPModelMeta) = meta.ncon == 0 && !has_bounds(meta)

"""
    linearly_constrained(nlp)
    linearly_constrained(meta)

Returns whether the problem's constraints are known to be all linear.
"""
linearly_constrained(meta::NLPModelMeta) = meta.nlin == meta.ncon > 0

"""
    equality_constrained(nlp)
    equality_constrained(meta)

Returns whether the problem's constraints are all equalities.
Unconstrained problems return false.
"""
equality_constrained(meta::NLPModelMeta) = length(meta.jfix) == meta.ncon > 0

"""
    inequality_constrained(nlp)
    inequality_constrained(meta)

Returns whether the problem's constraints are all inequalities.
Unconstrained problems return true.
"""
inequality_constrained(meta::NLPModelMeta) = meta.ncon > 0 && length(meta.jfix) == 0

for meth in (:has_bounds, :bound_constrained, :unconstrained,
    :linearly_constrained, :equality_constrained, :inequality_constrained)
  @eval $meth(nlp::AbstractNLPModel) = $meth(nlp.meta)
end
