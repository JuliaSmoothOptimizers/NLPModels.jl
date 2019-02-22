export QuasiNewtonModel, LBFGSModel, LSR1Model

abstract type QuasiNewtonModel <: AbstractNLPModel end

mutable struct LBFGSModel <: QuasiNewtonModel
  meta :: NLPModelMeta
  model :: AbstractNLPModel
  op :: LBFGSOperator
end

mutable struct LSR1Model <: QuasiNewtonModel
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

sum_counters(nlp :: QuasiNewtonModel) = sum_counters(nlp.model)

function increment!(nlp :: QuasiNewtonModel, s :: Symbol)
  increment!(nlp.model, s)
end

function reset!(nlp :: QuasiNewtonModel)
  reset!(nlp.model.counters)
  reset!(nlp.op)
  return nlp
end

# the following methods are not affected by the Hessian approximation
for meth in (:obj, :grad, :cons, :jac_coord, :jac)
  @eval $meth(nlp :: QuasiNewtonModel, x :: AbstractVector) = $meth(nlp.model, x)
end
for meth in (:grad!, :cons!, :jprod, :jtprod)
  @eval $meth(nlp :: QuasiNewtonModel, x :: AbstractVector, y :: AbstractVector) = $meth(nlp.model, x, y)
end
for meth in (:jprod!, :jtprod!)
  @eval $meth(nlp :: QuasiNewtonModel, x :: AbstractVector, y :: AbstractVector, z :: AbstractVector) = $meth(nlp.model, x, y, z)
end
jac_structure(nlp :: QuasiNewtonModel) = jac_structure(nlp.model)
jac_coord!(nlp :: QuasiNewtonModel, x :: AbstractVector, rows :: AbstractVector{<: Integer},
           cols :: AbstractVector{<: Integer}, vals :: AbstractVector) =
    jac_coord!(nlp.model, x, rows, cols, vals)

# the following methods are affected by the Hessian approximation
hess_op(nlp :: QuasiNewtonModel, x :: AbstractVector; kwargs...) = nlp.op
hprod(nlp :: QuasiNewtonModel, x :: AbstractVector, v :: AbstractVector; kwargs...) = nlp.op * v
function hprod!(nlp :: QuasiNewtonModel, x :: AbstractVector,
                v :: AbstractVector, Hv :: AbstractVector; kwargs...)
  Hv[1:nlp.meta.nvar] .= nlp.op * v
  return Hv
end

function push!(nlp :: QuasiNewtonModel, args...)
	push!(nlp.op, args...)
	return nlp
end

# not implemented: hess_structure, hess_coord, hess
