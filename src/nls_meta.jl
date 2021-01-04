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
  show(io, nls.meta, nls.nls_meta)
  show(io, nls.counters)
end

function lines_of_description(nm :: NLSMeta)
  V = [nm.nequ, nm.nlin, nm.nnln]
  S = ["All residuals", "linear", "nonlinear"]
  lines = lines_of_hist(S, V)
  push!(lines,
        sparsityline("nnzj", nm.nnzj, nm.nvar * nm.nequ),
        sparsityline("nnzh", nm.nnzh, nm.nvar * (nm.nvar + 1) / 2))

  return lines
end

function show(io :: IO, nm :: NLSMeta)
  lines = lines_of_description(nm)
  println(io, join(lines, "\n") * "\n")
end

function show(io :: IO, m :: NLPModelMeta, nm :: NLSMeta)
  println("  Problem name: $(m.name)")
  nlplines = lines_of_description(m)
  nlslines = lines_of_description(nm)
  append!(nlslines, repeat([""], length(nlplines) - length(nlslines)))
  lines = nlplines .* nlslines
  println(io, join(lines, "\n") * "\n")
end

function show(io :: IO, c :: NLSCounters)
  println(io, "  Counters:")
  F = setdiff(fieldnames(Counters) ∪ fieldnames(NLSCounters), [:counters])
  show_counters(io, c, F)
end