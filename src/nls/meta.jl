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
- `lin`: indices of linear constraints
- `nln`: indices of nonlinear constraints
"""
struct NLSMeta
  nequ::Int
  nvar::Int
  x0::Vector
  nnzj::Int  # Number of elements needed to store the nonzeros of the Jacobian of the residual
  nnzh::Int  # Number of elements needed to store the nonzeros of the sum of Hessians of the residuals

  nln::Vector{Int} # List of nonlinear residuals
  nnln::Int # = length(nln)
  lin::Vector{Int} # List of linear residuals
  nlin::Int # = length(lin)
end

function NLSMeta(
  nequ::Int,
  nvar::Int;
  x0::AbstractVector=zeros(nvar),
  nnzj=nequ * nvar,
  nnzh=div(nvar * (nvar + 1), 2),
  nln=1:nequ,
  lin=Int[],
)
  nnzj = max(0, nnzj)
  nnzh = max(0, nnzh)
  return NLSMeta(nequ, nvar, x0, nnzj, nnzh, nln, length(nln), lin, length(lin))
end

"""
    nls_meta(nls)

Returns the `nls_meta` structure of `nls`.
Use this instead of `nls.nls_meta` to handle models that have internal models.

For basic models `nls_meta(nls)` is defined as `nls.nls_meta`, but composite models might not keep `nls_meta` themselves, so they might specialize it to something like `nls.internal.nls_meta`.
"""
nls_meta(nls::AbstractNLSModel) = nls.nls_meta
