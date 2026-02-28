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
function lines_of_description(m::M) where {M <: AbstractNLPModelMeta}
  V = [
    length(get_ifree(m)),
    length(get_ilow(m)),
    length(get_iupp(m)),
    length(get_irng(m)),
    length(get_ifix(m)),
    length(get_iinf(m)),
  ]
  V = [sum(V); V]
  S = ["All variables", "free", "lower", "upper", "low/upp", "fixed", "infeas"]
  varlines = lines_of_hist(S, V)
  push!(varlines, sparsityline("nnzh", get_nnzh(m), get_nvar(m) * (get_nvar(m) + 1) / 2))

  V = [
    length(get_jfree(m)),
    length(get_jlow(m)),
    length(get_jupp(m)),
    length(get_jrng(m)),
    length(get_jfix(m)),
    length(get_jinf(m)),
  ]
  V = [sum(V); V]
  S = ["All constraints", "free", "lower", "upper", "low/upp", "fixed", "infeas"]
  conlines = lines_of_hist(S, V)

  append!(
    conlines,
    [
      histline("linear", get_nlin(m), get_ncon(m)),
      histline("nonlinear", get_nnln(m), get_ncon(m)),
      sparsityline("nnzj", get_nnzj(m), get_nvar(m) * get_ncon(m)),
    ],
  )

  if :lin_nnzj in fieldnames(M)
    push!(conlines, sparsityline("lin_nnzj", get_lin_nnzj(m), get_nlin(m) * get_nvar(m)))
  end
  if :nln_nnzj in fieldnames(M)
    push!(conlines, sparsityline("nln_nnzj", get_nln_nnzj(m), get_nnln(m) * get_nvar(m)))
  end

  append!(varlines, repeat([" "^length(varlines[1])], length(conlines) - length(varlines)))
  lines = varlines .* conlines

  return lines
end

function Base.show(io::IO, m::AbstractNLPModelMeta)
  println(io, "  Problem name: $(get_name(m))")
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
