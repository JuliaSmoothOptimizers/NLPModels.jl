export ModelOperator, update!

using FastClosures, LinearOperators
import LinearOperators.AbstractLinearOperator

mutable struct ModelOperator{T, I <: Integer, F, Ft, Fct, S, M <: AbstractNLPModel{T, S}} <:
               AbstractLinearOperator{T}
  x::S
  nlp::M
  nrow::I
  ncol::I
  symmetric::Bool
  hermitian::Bool
  prod!::F
  tprod!::Ft
  ctprod!::Fct
  nprod::I
  ntprod::I
  nctprod::I
  args5::Bool
  use_prod5!::Bool # true for 5-args mul! and for composite operators created with operators that use the 3-args mul!
  Mv5::S
  Mtu5::S
  allocated5::Bool # true for 5-args mul!, false for 3-args mul! until the vectors are allocated
end

function ModelOperator(
  x::S,
  nlp::M,
  nrow::I,
  ncol::I,
  symmetric::Bool,
  hermitian::Bool,
  prod!::F,
  tprod!::Ft,
  ctprod!::Fct,
  nprod::I,
  ntprod::I,
  nctprod::I;
) where {T, I <: Integer, F, Ft, Fct, S, M <: AbstractNLPModel{T, S}}
  Mv5, Mtu5 = S(undef, 0), S(undef, 0)
  nargs = first(methods(prod!)).nargs - 1
  args5 = (nargs == 4)
  (args5 == false) || (nargs != 2) || throw(LinearOperatorException("Invalid number of arguments"))
  allocated5 = args5 ? true : false
  use_prod5! = args5 ? true : false
  return ModelOperator{T, I, F, Ft, Fct, S, M}(
    x,
    nlp,
    nrow,
    ncol,
    symmetric,
    hermitian,
    prod!,
    tprod!,
    ctprod!,
    nprod,
    ntprod,
    nctprod,
    args5,
    use_prod5!,
    Mv5,
    Mtu5,
    allocated5,
  )
end

ModelOperator(
  x::S,
  nlp::M,
  nrow::I,
  ncol::I,
  symmetric::Bool,
  hermitian::Bool,
  prod!,
  tprod!,
  ctprod!,
) where {T, I <: Integer, S, M <: AbstractNLPModel{T, S}} =
  ModelOperator(x, nlp, nrow, ncol, symmetric, hermitian, prod!, tprod!, ctprod!, 0, 0, 0)

function update!(op::ModelOperator, x)
  op.x .= x
end

function HprodOperator!(
  nlp::M,
  x::S,
  Hv::S;
  obj_weight::Real = one(T),
) where {T, S, M <: AbstractNLPModel{T, S}}
  prod! = @closure (res, v, α, β) -> begin
    hprod!(nlp, x, v, Hv; obj_weight = obj_weight)
    if β == 0
      @. res = α * Hv
    else
      @. res = α * Hv + β * res
    end
    return res
  end

  return ModelOperator(x, nlp, nlp.meta.nvar, nlp.meta.nvar, true, true, prod!, prod!, prod!)
end

function HprodOperator!(
  nlp::M,
  Hv::S;
  obj_weight::Real = one(T),
) where {T, S, M <: AbstractNLPModel{T, S}}
  x = copy(nlp.meta.x0)
  HprodOperator!(nlp, x, Hv; obj_weight = obj_weight)
end

function HprodOperator(nlp::M; obj_weight::Real = one(T)) where {T, S, M <: AbstractNLPModel{T, S}}
  x = copy(nlp.meta.x0)
  Hv = S(undef, nlp.meta.nvar)
  HprodOperator!(nlp, x, Hv; obj_weight = obj_weight)
end
