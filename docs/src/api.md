# API

```@eval
using NLPModels
using Base.Markdown

s = []
mtds = [obj, grad, grad!, cons, cons!, jac_coord, jac, jprod, jprod!,
  jtprod, jtprod!, hess_coord, hess, hprod, hprod!, NLPtoMPB, reset!]
models = [("SimpleNLPModel","simple_model.jl"),
  ("JuMPNLPModel","jump_model.jl"), ("SlackModel","slack_model.jl")]
path = Pkg.dir("NLPModels", "src")
files = map(x->readall(open(joinpath(path, x[2]))), models)

for i = 1:length(mtds)
  name = split(string(mtds[i]), ".")[2]
  push!(s, md"### $name")
  push!(s, @doc mtds[i])
  sout = []
  for j = 1:length(files)
    if contains(files[j], "function $name")
      md = models[j][1]
      push!(sout, "[$md](/models/#$md)")
    end
  end
  push!(s, "Implemented by " * join(sout, ", "))
end

s
```
