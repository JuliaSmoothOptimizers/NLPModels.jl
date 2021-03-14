function Base.show(io :: IO, nls :: AbstractNLSModel)
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

function Base.show(io :: IO, nm :: NLSMeta)
  lines = lines_of_description(nm)
  println(io, join(lines, "\n") * "\n")
end

function Base.show(io :: IO, m :: NLPModelMeta, nm :: NLSMeta)
  println(io, "  Problem name: $(m.name)")
  nlplines = lines_of_description(m)
  nlslines = lines_of_description(nm)
  append!(nlslines, repeat([""], length(nlplines) - length(nlslines)))
  lines = nlplines .* nlslines
  println(io, join(lines, "\n") * "\n")
end

function Base.show(io :: IO, c :: NLSCounters)
  println(io, "  Counters:")
  F = setdiff(fieldnames(Counters) ∪ fieldnames(NLSCounters), [:counters])
  show_counters(io, c, F)
end
