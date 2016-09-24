# Base type for an optimization model.
abstract AbstractNLPModel

type Counters
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
abstract AbstractNLPModelMeta

immutable NLPModelMeta <: AbstractNLPModelMeta

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
  x0   :: Array{Float64,1}  # initial guess
  lvar :: Array{Float64,1}  # vector of lower bounds
  uvar :: Array{Float64,1}  # vector of upper bounds

  ifix  :: Array{Int,1}     # indices of fixed variables
  ilow  :: Array{Int,1}     # indices of variables with lower bound only
  iupp  :: Array{Int,1}     # indices of variables with upper bound only
  irng  :: Array{Int,1}     # indices of variables with lower and upper bound (range)
  ifree :: Array{Int,1}     # indices of free variables
  iinf  :: Array{Int,1}     # indices of infeasible bounds

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
  y0   :: Array{Float64,1}  # initial Lagrange multipliers
  lcon :: Array{Float64,1}  # vector of constraint lower bounds
  ucon :: Array{Float64,1}  # vector of constraint upper bounds

  jfix  :: Array{Int,1}     # indices of equality constraints
  jlow  :: Array{Int,1}     # indices of constraints of the form c(x) ≥ cl
  jupp  :: Array{Int,1}     # indices of constraints of the form c(x) ≤ cu
  jrng  :: Array{Int,1}     # indices of constraints of the form cl ≤ c(x) ≤ cu
  jfree :: Array{Int,1}     # indices of "free" constraints (there shouldn't be any)
  jinf  :: Array{Int,1}     # indices of the visibly infeasible constraints

  nnzo :: Int               # number of nonzeros in all objectives gradients
  nnzj :: Int               # number of nonzeros in the sparse Jacobian
  nnzh :: Int               # number of nonzeros in the sparse Hessian

  nlin  :: Int              # number of linear constraints
  nnln  :: Int              # number of nonlinear general constraints
  nnnet :: Int              # number of nonlinear network constraints
  nlnet :: Int              # number of linear network constraints

  lin   :: Array{Int,1}     # indices of linear constraints
  nln   :: Array{Int,1}     # indices of nonlinear constraints
  nnet  :: Array{Int,1}     # indices of nonlinear network constraints
  lnet  :: Array{Int,1}     # indices of linear network constraints

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
                        lin=collect([]),
                        nln=1:ncon,
                        nnet=collect([]),
                        lnet=collect([]),
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

    ifix  = find(lvar .== uvar);
    ilow  = find((lvar .> -Inf) & (uvar .== Inf));
    iupp  = find((lvar .== -Inf) & (uvar .< Inf));
    irng  = find((lvar .> -Inf) & (uvar .< Inf) & (lvar .< uvar));
    ifree = find((lvar .== -Inf) & (uvar .== Inf));
    iinf  = find(lvar .> uvar);

    jfix  = find(lcon .== ucon);
    jlow  = find((lcon .> -Inf) & (ucon .== Inf));
    jupp  = find((lcon .== -Inf) & (ucon .< Inf));
    jrng  = find((lcon .> -Inf) & (ucon .< Inf) & (lcon .< ucon));
    jfree = find((lcon .== -Inf) & (ucon .== Inf));
    jinf  = find(lcon .> ucon);

    nnzj = max(0, min(nnzj, nvar * ncon));
    nnzh = max(0, min(nnzh, nvar * nvar));

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

import Base.show, Base.print
function show(io :: IO, nlp :: NLPModelMeta)
  s  = nlp.minimize ? @sprintf("Minimization ") : @sprintf("Maximization ")
  s *= @sprintf("problem %s\n", nlp.name)
  s *= @sprintf("nvar = %d, ncon = %d (%d linear)\n", nlp.nvar, nlp.ncon, nlp.nlin)
  print(io, s)
end

function print(io :: IO, nlp :: NLPModelMeta)
  nlp.minimize ? @printf("Minimization ") : @printf("Maximization ")
  @printf("problem %s\n", nlp.name)
  @printf("nvar = %d, ncon = %d (%d linear)\n", nlp.nvar, nlp.ncon, nlp.nlin)
  @printf("lvar = "); display(nlp.lvar'); @printf("\n")
  @printf("uvar = "); display(nlp.uvar'); @printf("\n")
  @printf("lcon = "); display(nlp.lcon'); @printf("\n")
  @printf("ucon = "); display(nlp.ucon'); @printf("\n")
  @printf("x0 = ");   display(nlp.x0'); @printf("\n")
  @printf("y0 = ");   display(nlp.y0'); @printf("\n")
  @printf("nnzh = %d\n", nlp.nnzh);
  @printf("nnzj = %d\n", nlp.nnzj);
  if nlp.nlin > 0
    @printf("linear constraints:    "); display(nlp.lin'); @printf("\n");
  end
  if nlp.nnln > 0
    @printf("nonlinear constraints: "); display(nlp.nln'); @printf("\n");
  end
  if nlp.nlnet > 0
    @printf("linear network constraints:   "); display(nlp.lnet'); @printf("\n");
  end
  if nlp.nnnet > 0
    @printf("nonlinear network constraints:   "); display(nlp.nnet'); @printf("\n");
  end
end
