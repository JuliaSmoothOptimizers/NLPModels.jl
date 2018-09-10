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
end

function NLSMeta(nequ :: Int, nvar :: Int;
                 x0 :: AbstractVector = zeros(nvar))
  return NLSMeta(nequ, nvar, x0)
end
