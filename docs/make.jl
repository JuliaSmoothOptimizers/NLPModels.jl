using Documenter, NLPModels

makedocs(
  modules = [NLPModels]
)

deploydocs(deps = Deps.pip("mkdocs", "python-markdown-math"),
  repo = "github.com/JuliaSmoothOptimizers/NLPModels.jl.git",
  julia = "release",
  latest = "develop"
)
