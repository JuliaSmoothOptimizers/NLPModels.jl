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

import Base.show
function show(io :: IO, nls :: AbstractNLSModel)
  show_header(io, nls)
  show(io, nls.meta)
  show(io, nls.nls_meta)
  show(io, nls.counters)
end

function show(io :: IO, nm :: NLSMeta)
  sep(a) = (a == "" ? " " : "…")^(18-length(a))
  for (a,b) in [("Total residuals", nm.nequ),
                ("  linear", nm.nlin),
                ("  nonlinear", nm.nnln),
                ("  nnzj", nm.nnzj),
                ("  nnzh", nm.nnzh)]
    @printf(io, "  %s%s%-6s\n", a, sep(a), b)
  end
end

function show(io :: IO, c :: NLSCounters)
  println(io, "  Counters:")
  k = 0
  sep(s) = (s == "" ? " " : "…")^(17-length(s))
  for f in fieldnames(Counters) ∪ fieldnames(NLSCounters)
    f == :counters && continue
    s = string(f)[7:end]
    @printf(io, "    %s%s%-6s", s, sep(s), getproperty(c, f))
    k += 1
    k % 4 == 0 && println(io, "")
  end
end
