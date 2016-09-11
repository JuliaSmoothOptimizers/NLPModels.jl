export QuasiNewtonModel, LBFGSModel, LSR1Model,
       reset!,
       obj, grad, grad!,
       cons, cons!, jac_coord, jac, jprod, jprod!, jtprod, jtprod!,
       hprod, hess_op,
       push!

abstract QuasiNewtonModel <: AbstractNLPModel

type LBFGSModel <: QuasiNewtonModel
  meta :: NLPModelMeta
  model :: AbstractNLPModel
  op :: LBFGSOperator
end

type LSR1Model <: QuasiNewtonModel
  meta :: NLPModelMeta
  model :: AbstractNLPModel
  op :: LSR1Operator
end

"Construct a `LBFGSModel` from another type of model."
function LBFGSModel(nlp :: AbstractNLPModel; memory :: Int=5)
  op = LBFGSOperator(nlp.meta.nvar, memory)
  return LBFGSModel(nlp.meta, nlp, op)
end

"Construct a `LSR1Model` from another type of nlp."
function LSR1Model(nlp :: AbstractNLPModel; memory :: Int=5)
  op = LSR1Operator(nlp.meta.nvar, memory)
  return LSR1Model(nlp.meta, nlp, op)
end

# retrieve counters from underlying model
for counter in fieldnames(Counters)
  @eval begin
    $counter(nlp :: QuasiNewtonModel) = $counter(nlp.model)
    export $counter
  end
end

function reset!(nlp :: QuasiNewtonModel)
  reset!(nlp.model.counters)
  reset!(nlp.op)
  return nlp
end

# the following methods are not affected by the Hessian approximation
for meth in (:obj, :grad, :grad!, :cons, :cons!, :jac_coord, :jac, :jprod, :jprod!, :jtprod, :jtprod!)
  @eval begin
    $meth(nlp :: QuasiNewtonModel, args...) = $meth(nlp.model, args...)
  end
end

# the following methods are affected by the Hessian approximation
hess_op(nlp :: QuasiNewtonModel, x :: Vector{Float64}; kwargs...) = nlp.op
hprod(nlp :: QuasiNewtonModel, x :: Vector{Float64}, v :: Vector{Float64}; kwargs...) = nlp.op * v

function push!(nlp :: QuasiNewtonModel, args...)
	push!(nlp.op, args...)
	return nlp
end

# not implemented: hess_coord, hess, hprod!
