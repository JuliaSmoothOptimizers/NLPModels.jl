export ParametricNLPModelMeta

struct ParametricNLPModelMeta
    nparam::Int
    nnzj::Int      # ∇ₚ g
    nnzh::Int      # ∇ₚ (∇ₓ L)
    nnzjlcon::Int  # ∇ₚ lcon
    nnzjucon::Int  # ∇ₚ ucon
    nnzjlvar::Int  # ∇ₚ lvar
    nnzjuvar::Int  # ∇ₚ uvar
    grad_param_available::Bool
    jac_param_available::Bool
    hess_param_available::Bool
    jpprod_available::Bool
    jptprod_available::Bool
    hpprod_available::Bool
    hptprod_available::Bool
    lcon_jac_available::Bool
    ucon_jac_available::Bool
    lvar_jac_available::Bool
    uvar_jac_available::Bool
    lcon_jpprod_available::Bool
    ucon_jpprod_available::Bool
    lvar_jpprod_available::Bool
    uvar_jpprod_available::Bool
    lcon_jptprod_available::Bool
    ucon_jptprod_available::Bool
    lvar_jptprod_available::Bool
    uvar_jptprod_available::Bool
end

for field in fieldnames(ParametricNLPModelMeta)
  meth = Symbol("get_", field)
  @eval begin
    $meth(meta::ParametricNLPModelMeta) = getproperty(meta, $(QuoteNode(field)))
  end
  @eval $meth(bnlp::AbstractNLPModel) = $meth(bnlp.meta)
  @eval export $meth
end

function ParametricNLPModelMeta(;
    nparam::Int = 0,
    nnzj::Int = 0,
    nnzh::Int = 0,
    nnzjlcon::Int = 0,
    nnzjucon::Int = 0,
    nnzjlvar::Int = 0,
    nnzjuvar::Int = 0,
    grad_param_available::Bool = false,
    jac_param_available::Bool = false,
    hess_param_available::Bool = false,
    jpprod_available::Bool = false,
    jptprod_available::Bool = false,
    hpprod_available::Bool = false,
    hptprod_available::Bool = false,
    lcon_jac_available::Bool = false,
    ucon_jac_available::Bool = false,
    lvar_jac_available::Bool = false,
    uvar_jac_available::Bool = false,
    lcon_jpprod_available::Bool = false,
    ucon_jpprod_available::Bool = false,
    lvar_jpprod_available::Bool = false,
    uvar_jpprod_available::Bool = false,
    lcon_jptprod_available::Bool = false,
    ucon_jptprod_available::Bool = false,
    lvar_jptprod_available::Bool = false,
    uvar_jptprod_available::Bool = false,
)
    return ParametricNLPModelMeta(
        nparam, nnzj, nnzh, nnzjlcon, nnzjucon, nnzjlvar, nnzjuvar,
        grad_param_available, jac_param_available, hess_param_available,
        jpprod_available, jptprod_available, hpprod_available, hptprod_available,
        lcon_jac_available, ucon_jac_available, lvar_jac_available, uvar_jac_available,
        lcon_jpprod_available, ucon_jpprod_available, lvar_jpprod_available, uvar_jpprod_available,
        lcon_jptprod_available, ucon_jptprod_available, lvar_jptprod_available, uvar_jptprod_available,
    )
end