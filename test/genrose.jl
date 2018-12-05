# Generalized Rosenbrock function.
#
#   Source:
#   Y.-W. Shang and Y.-H. Qiu,
#   A note on the extended Rosenbrock function,
#   Evolutionary Computation, 14(1):119–126, 2006.
#
# Shang and Qiu claim the "extended" Rosenbrock function
# previously appeared in
#
#   K. A. de Jong,
#   An analysis of the behavior of a class of genetic
#   adaptive systems,
#   PhD Thesis, University of Michigan, Ann Arbor,
#   Michigan, 1975,
#   (http://hdl.handle.net/2027.42/4507)
#
# but I could not find it there, and in
#
#   D. E. Goldberg,
#   Genetic algorithms in search, optimization and
#   machine learning,
#   Reading, Massachusetts: Addison-Wesley, 1989,
#
# but I don't have access to that book.
#
# This unconstrained problem is analyzed in
#
#   S. Kok and C. Sandrock,
#   Locating and Characterizing the Stationary Points of
#   the Extended Rosenbrock Function,
#   Evolutionary Computation 17, 2009.
#   https://dx.doi.org/10.1162%2Fevco.2009.17.3.437
#
#   classification SUR2-AN-V-0
#
# D. Orban, Montreal, 08/2015.

"Generalized Rosenbrock model in size `n`"
function genrose_autodiff(n :: Int=10)

  n < 2 && error("genrose: number of variables must be ≥ 2")

  x0 = [i/(n+1) for i = 1:n]
  f(x::AbstractVector) = begin
    s = 1.0
    for i = 1:n-1
      s += 100 * (x[i+1] - x[i]^2)^2 + (x[i] - 1)^2
    end
    return s
  end

  return ADNLPModel(f, x0)
end

mutable struct GENROSE <: AbstractNLPModel
  meta :: NLPModelMeta
  σnls :: Float64
  counters :: Counters
end

function GENROSE(n :: Int=10)
  meta = NLPModelMeta(n, nobjs=0, nlsequ=2n-1, llsrows=0, x0 = [i/(n+1) for i = 1:n])
  return GENROSE(meta, 2.0, Counters())
end

function NLPModels.residual!(nlp :: GENROSE, x :: AbstractVector, Fx :: AbstractVector)
  increment!(nlp, :neval_residual)
  n = nvar(nlp)
  for i = 1:n-1
    Fx[i] = 10 * (x[i+1] - x[i]^2)
    Fx[n+i-1] = x[i] - 1
  end
  Fx[2n-1] = 1.0
  return Fx
end

function NLPModels.jac_residual(nlp :: GENROSE, x :: AbstractVector)
  increment!(nlp, :neval_jac_residual)
  n = nvar(nlp)
  Jx = zeros(2n-1, n)
  for i = 1:n-1
    Jx[i,i] = -20.0 * x[i]
    Jx[i,i+1] = 10.0
    Jx[n+i-1,i] = 1.0
  end
  return Jx
end

function NLPModels.jprod_residual!(nlp :: GENROSE, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jprod_residual)
  nlp.counters.neval_jac_residual -= 1
  Jx = jac_residual(nlp, x)
  Jv .= Jx * v
  return Jv
end

function NLPModels.jtprod_residual!(nlp :: GENROSE, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jprod_residual)
  nlp.counters.neval_jac_residual -= 1
  Jx = jac_residual(nlp, x)
  Jtv .= Jx' * v
  return Jtv
end

function NLPModels.hess_residual(nlp :: GENROSE, i :: Int, x :: AbstractVector)
  increment!(nlp, :neval_hess_residual)
  n = nvar(nlp)
  Hx = spzeros(n, n)
  if 1 ≤ i ≤ n - 1
    Hx[i,i] = -20.0
  end
  return Hx
end

function NLPModels.hprod_residual!(nlp :: GENROSE, i :: Int, x :: AbstractVector, v :: AbstractVector, Hiv :: AbstractVector)
  increment!(nlp, :neval_hprod_residual)
  n = nvar(nlp)
  fill!(Hiv, 0.0)
  if 1 ≤ i ≤ n - 1
    Hiv[i] = -20 * v[i]
  end
  return Hiv
end
