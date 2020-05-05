export NLSMeta

# The problem is
#
#    min    ¹/₂‖F(x)‖²
#
# where F:ℜⁿ→ℜᵐ. `n` = `nvar`, `m` = `nequ`.
#
# TODO: Extend

struct NLSMeta
  nequ :: Int
  nvar :: Int
  x0 :: Vector
  nnzj :: Int  # Number of elements needed to store the nonzeros of the Jacobian of the residual
  nnzh :: Int  # Number of elements needed to store the nonzeros of the sum of Hessians of the residuals

  nln :: Vector{Int} # List of nonlinear residuals
  nnln :: Int # = length(nln)
  lin :: Vector{Int} # List of linear residuals
  nlin :: Int # = length(lin)
end

function NLSMeta(nequ :: Int, nvar :: Int;
                 x0 :: AbstractVector = zeros(nvar),
                 nnzj=nequ * nvar,
                 nnzh=div(nvar * (nvar + 1), 2),
                 nln=1:nequ,
                 lin=Int[],
                )
  nnzj = max(0, nnzj)
  nnzh = max(0, nnzh)
  return NLSMeta(nequ, nvar, x0, nnzj, nnzh, nln, length(nln), lin, length(lin))
end
