export NLSMeta, nls_meta

"""
    NLSMeta

Base type for metadata related to a nonlinear least-squares model.

---

    NLSMeta(nequ, nvar; kwargs...)

Create a `NLSMeta` with `nequ` equations and `nvar` variables.
The following keyword arguments are accepted:
- `x0`: initial guess
- `nnzj`: number of elements needed to store the nonzeros of the Jacobian of the residual
- `nnzh`: number of elements needed to store the nonzeros of the sum of Hessians of the residuals
- `lin`: indices of linear residuals
- `jac_residual_available`: indicates whether the sparse Jacobian of the residuals is available
- `hess_residual_available`: indicates whether the sum of the sparse Hessians of the residuals is available
- `jprod_residual_available`: indicates whether the Jacobian-vector product for the residuals is available
- `jtprod_residual_available`: indicates whether the transpose Jacobian-vector product for the residuals is available
- `hprod_residual_available`: indicates whether the Hessian-vector product for each residual is available

`NLSMeta` also contains the following attributes, which are computed from the variables above:
- `nequ`: size of the residual
- `nvar`: number of variables
- `nln`: indices of nonlinear residuals
- `nnln`: number of nonlinear general residuals
- `nlin`: number of linear residuals
"""
struct NLSMeta{T, S}
  nequ::Int
  nvar::Int
  x0::S
  nnzj::Int  # Number of elements needed to store the nonzeros of the Jacobian of the residual
  nnzh::Int  # Number of elements needed to store the nonzeros of the sum of Hessians of the residuals

  nln::Vector{Int} # List of nonlinear residuals
  nnln::Int # = length(nln)
  lin::Vector{Int} # List of linear residuals
  nlin::Int # = length(lin)

  jac_residual_available::Bool
  hess_residual_available::Bool
  jprod_residual_available::Bool
  jtprod_residual_available::Bool
  hprod_residual_available::Bool

  function NLSMeta{T, S}(
    nequ::Int,
    nvar::Int;
    x0::S = fill!(S(undef, nvar), zero(T)),
    nnzj = nequ * nvar,
    nnzh = div(nvar * (nvar + 1), 2),
    lin = Int[],
    jac_residual_available::Bool = true,
    hess_residual_available::Bool = true,
    jprod_residual_available::Bool = true,
    jtprod_residual_available::Bool = true,
    hprod_residual_available::Bool = true,
  ) where {T, S}
    nnzj = max(0, nnzj)
    nnzh = max(0, nnzh)

    nln = setdiff(1:nequ, lin)
    nlin = length(lin)
    nnln = length(nln)

    return new{T, S}(nequ, nvar, x0, nnzj, nnzh, nln, nnln, lin, nlin,
                     jac_residual_available, hess_residual_available,
                     jprod_residual_available, jtprod_residual_available,
                     hprod_residual_available)
  end
end

NLSMeta(nequ::Int, nvar::Int; x0::S = zeros(nvar), kwargs...) where {S} =
  NLSMeta{eltype(S), S}(nequ, nvar; x0 = x0, kwargs...)

"""
    nls_meta(nls)

Returns the `nls_meta` structure of `nls`.
Use this instead of `nls.nls_meta` to handle models that have internal models.

For basic models `nls_meta(nls)` is defined as `nls.nls_meta`, but composite models might not keep `nls_meta` themselves, so they might specialize it to something like `nls.internal.nls_meta`.
"""
nls_meta(nls::AbstractNLSModel) = nls.nls_meta
