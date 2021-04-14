export @default_nlscounters

"""
    @default_nlscounters Model inner

Define functions relating NLS counters of `Model` to NLS counters of `Model.inner`.
"""
macro default_nlscounters(Model, inner)
  ex = Expr(:block)
  for foo in fieldnames(NLSCounters) ∪ [:sum_counters, :reset!]
    foo == :counters && continue
    push!(ex.args, :(NLPModels.$foo(nlp::$(esc(Model))) = $foo(nlp.$inner)))
  end
  push!(ex.args, :(NLPModels.increment!(nlp::$(esc(Model)), s::Symbol) = increment!(nlp.$inner, s)))
  push!(ex.args, :(NLPModels.decrement!(nlp::$(esc(Model)), s::Symbol) = decrement!(nlp.$inner, s)))
  ex
end
