export show_header

"""
    show_header(io, nlp)

Show a header for the specific `nlp` type.
Should be imported and defined for every model implementing the NLPModels API.
"""
show_header(io::IO, nlp::AbstractNLPModel) = println(io, typeof(nlp))

function Base.show(io::IO, nlp::AbstractNLPModel)
  show_header(io, nlp)
  show(io, nlp.meta)
  show(io, nlp.counters)
end

"""
    histline(s, v, maxv)

Return a string of the form

    ______NAME______: ████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 5

where:
- `______NAME______` is `s` with padding to the left and length 16.
- And the symbols █ and ⋅ fill 20 characters in the proportion of `v / maxv` to █ and the rest to ⋅.
- The number `5` is v.
"""
function histline(s, v, maxv)
  @assert 0 ≤ v ≤ maxv
  λ = maxv == 0 ? 0 : ceil(Int, 20 * v / maxv)
  return @sprintf("%16s: %s %-6s", s, "█"^λ * "⋅"^(20 - λ), v)
end

"""
    sparsityline(s, v, maxv)

Return a string of the form

    ______NAME______: ( 80.00% sparsity)   5

where:
- `______NAME______` is `s` with padding to the left and length 16.
- The sparsity value is given by `v / maxv`.
- The number `5` is v.
"""
function sparsityline(s, v, maxv)
  if maxv == 0
    @sprintf("%16s: (------%% sparsity)   %-6s", s, " ")
  else
    @sprintf("%16s: (%6.2f%% sparsity)   %-6s", s, 100 * (1 - v / maxv), v)
  end
end

"""
    lines_of_hist(S, V)

Return a vector of `histline(s, v, maxv)`s using pairs of `s` in `S` and `v` in `V`. `maxv` is given by the maximum of `V`.
"""
function lines_of_hist(S, V)
  maxv = maximum(V)
  lines = histline.(S, V, maxv)
  return lines
end

"""
    lines_of_description(meta)

Describe `meta` for the `show` function.
"""
function lines_of_description(m::AbstractNLPModelMeta)
  V = [
    length(m.ifree),
    length(m.ilow),
    length(m.iupp),
    length(m.irng),
    length(m.ifix),
    length(m.iinf),
  ]
  V = [sum(V); V]
  S = ["All variables", "free", "lower", "upper", "low/upp", "fixed", "infeas"]
  varlines = lines_of_hist(S, V)
  push!(varlines, sparsityline("nnzh", m.nnzh, m.nvar * (m.nvar + 1) / 2))

  V = [
    length(m.jfree),
    length(m.jlow),
    length(m.jupp),
    length(m.jrng),
    length(m.jfix),
    length(m.jinf),
  ]
  V = [sum(V); V]
  S = ["All constraints", "free", "lower", "upper", "low/upp", "fixed", "infeas"]
  conlines = lines_of_hist(S, V)

  append!(
    conlines,
    [
      histline("linear", m.nlin, m.ncon),
      histline("nonlinear", m.nnln, m.ncon),
      sparsityline("nnzj", m.nnzj, m.nvar * m.ncon),
    ],
  )

  if :lin_nnzj in fieldnames(typeof(m))
    push!(conlines, sparsityline("lin_nnzj", m.lin_nnzj, m.nlin * m.nvar))
  end

  if :nln_nnzj in fieldnames(typeof(m))
    push!(conlines, sparsityline("nln_nnzj", m.nln_nnzj, m.nnln * m.nvar))
  end

  maxlen = max(length(varlines), length(conlines))
  while length(varlines) < maxlen
    push!(varlines, " "^length(varlines[1]))
  end
  while length(conlines) < maxlen
    push!(conlines, " "^length(conlines[1]))
  end

  return varlines .* conlines
end

function Base.show(io::IO, m::AbstractNLPModelMeta)
  println(io, "  Problem name: $(m.name)")
  lines = lines_of_description(m)
  println(io, join(lines, "\n") * "\n")
end

"""
    show_counters(io, counters, fields)

Show the `fields` of the struct `counters`.
"""
function show_counters(io::IO, c, F)
  V = (getproperty(c, f) for f in F)
  S = (string(f)[7:end] for f in F)
  lines = lines_of_hist(S, V)
  n = length(lines)
  for i = 1:3:length(lines)
    idx = i:min(n, i + 2)
    println(io, join(lines[idx], ""))
  end
end

function Base.show(io::IO, c::Counters)
  println(io, "  Counters:")
  F = fieldnames(Counters)
  show_counters(io, c, F)
end
