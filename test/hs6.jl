using NLPModels: increment!

#Problem 6 in the Hock-Schittkowski suite
mutable struct HS6 <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
end

mutable struct HS6NLS <: AbstractNLSModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  counters :: NLSCounters
end

function HS6()
  meta = NLPModelMeta(2, ncon=1, nnzh=1, nnzj=2, x0=[-1.2; 1.0], lcon=[0.0], ucon=[0.0], name="hs6")

  return HS6(meta, Counters())
end

function HS6NLS()
  meta = NLPModelMeta(2, ncon=1, nnzh=1, nnzj=2, x0=[-1.2; 1.0], lcon=[0.0], ucon=[0.0], name="hs6nls")
  nls_meta = NLSMeta(1, 2, nnzj=1, nnzh=0)

  return HS6NLS(meta, nls_meta, NLSCounters())
end

const HS6s = Union{HS6,HS6NLS}

function NLPModels.obj(nlp :: HS6s, x :: AbstractVector)
  increment!(nlp, :neval_obj)
  return (1 - x[1])^2
end

function NLPModels.grad!(nlp :: HS6s, x :: AbstractVector, gx :: AbstractVector)
  increment!(nlp, :neval_grad)
  gx .= [2 * (x[1] - 1); 0.0]
  return gx
end

function NLPModels.hess(nlp :: HS6s, x :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hess)
  w = length(y) > 0 ? y[1] : 0.0
  return [2.0 * obj_weight - 20 * w   0.0; 0.0 0.0]
end

function NLPModels.hess_structure!(nlp :: HS6s, rows :: AbstractVector{Int}, cols :: AbstractVector{Int})
  rows[1] = 1
  cols[1] = 1
  return rows, cols
end

function NLPModels.hess_coord!(nlp :: HS6s, x :: AbstractVector, rows :: AbstractVector{Int}, cols :: AbstractVector{Int}, vals :: AbstractVector; obj_weight=1.0, y=AbstractVector[])
  increment!(nlp, :neval_hess)
  vals[1] = 2.0 * obj_weight + (length(y) > 0 ? - 20 * y[1] : 0.0)
  return rows, cols, vals
end


function NLPModels.hess_coord(nlp :: HS6s, x :: AbstractVector; obj_weight :: Real=1.0, y::AbstractVector=Float64[])
  increment!(nlp, :neval_hess)
  if length(y) > 0
    return ([1], [1], [2.0 * obj_weight - 20 * y[1]])
  else
    return ([1], [1], [2.0 * obj_weight])
  end
end

function NLPModels.hprod!(nlp :: HS6s, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hprod)
  w = length(y) > 0 ? y[1] : 0.0
  Hv .= [(2.0 * obj_weight - 20 * w) * v[1]; 0.0]
  return Hv
end

function NLPModels.cons!(nlp :: HS6s, x :: AbstractVector, cx :: AbstractVector)
  increment!(nlp, :neval_cons)
  cx[1] = 10 * (x[2] - x[1]^2)
  return cx
end

function NLPModels.jac(nlp :: HS6s, x :: AbstractVector)
  increment!(nlp, :neval_jac)
  return [-20 * x[1]  10.0]
end

function NLPModels.jac_structure!(nlp :: HS6s, rows :: AbstractVector{Int}, cols :: AbstractVector{Int})
  rows[1:2] .= [1, 1]
  cols[1:2] .= [1, 2]
  return rows, cols
end

function NLPModels.jac_coord!(nlp :: HS6s, x :: AbstractVector, rows :: AbstractVector{Int}, cols :: AbstractVector{Int}, vals :: AbstractVector)
  increment!(nlp, :neval_jac)
  vals[1] = -20 * x[1]
  vals[2] = 10.0
  return rows, cols, vals
end

function NLPModels.jac_coord(nlp :: HS6s, x :: AbstractVector)
  increment!(nlp, :neval_jac)
  return ([1, 1], [1, 2], [-20 * x[1], 10.0])
end

function NLPModels.jprod!(nlp :: HS6s, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jprod)
  Jv .= [-20 * x[1] * v[1] + 10 * v[2]]
  return Jv
end

function NLPModels.jtprod!(nlp :: HS6s, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod)
  Jtv .= [-20 * x[1]; 10] * v[1]
  return Jtv
end

function NLPModels.residual!(nlp :: HS6NLS, x :: AbstractVector, Fx :: AbstractVector)
  increment!(nlp, :neval_residual)
  Fx[1] = (1 - x[1]) * sqrt(eltype(x)(2))
  return Fx
end

function NLPModels.jac_residual(nlp :: HS6NLS, x :: AbstractVector)
  increment!(nlp, :neval_jac_residual)
  return eltype(x)[-sqrt(eltype(x)(2))  0.0]
end

function NLPModels.jac_structure_residual!(nlp :: HS6NLS, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  rows[1] = 1
  cols[1] = 1
  return rows, cols
end

function NLPModels.jac_coord_residual!(nlp :: HS6NLS, x :: AbstractVector, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer}, vals :: AbstractVector)
  rows[1] = 1
  cols[1] = 1
  vals[1] = -sqrt(2*one(eltype(x)))
  return rows, cols, vals
end

function NLPModels.jprod_residual!(nlp :: HS6NLS, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jprod_residual)
  Jv[1] = -v[1] * sqrt(eltype(x)(2))
  return Jv
end

function NLPModels.jtprod_residual!(nlp :: HS6NLS, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod_residual)
  Jtv .= [-v[1]; zero(eltype(x))] * sqrt(eltype(x)(2))
  return Jtv
end

function NLPModels.hess_residual(nlp :: HS6NLS, x :: AbstractVector, v :: AbstractVector)
  increment!(nlp, :neval_hess_residual)
  return zeros(2, 2)
end

function NLPModels.hess_structure_residual!(nlp :: HS6NLS, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  return rows, cols
end

function NLPModels.hess_coord_residual!(nlp :: HS6NLS, x :: AbstractVector, v :: AbstractVector, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer}, vals :: AbstractVector)
  return rows, cols, vals
end

function NLPModels.jth_hess_residual(nlp :: HS6NLS, x :: AbstractVector, i :: Int)
  return zeros(2, 2)
end

function NLPModels.hprod_residual!(nlp :: HS6NLS, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  Hiv .= zero(eltype(x))
  return Hiv
end
