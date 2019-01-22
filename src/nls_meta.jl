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
  nnzj :: Int
  nnzh :: Int
end

function NLSMeta(nequ :: Int, nvar :: Int;
                 x0 :: AbstractVector = zeros(nvar),
                 nnzj=nequ * nvar,
                 nnzh=div(nvar * (nvar + 1), 2)
                )
  nnzj = max(0, min(nnzj, nequ * nvar))
  nnzh = max(0, min(nnzh, div(nvar * (nvar + 1), 2)))
  return NLSMeta(nequ, nvar, x0, nnzj, nnzh)
end
