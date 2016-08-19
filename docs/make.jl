using Documenter, NLPModels

makedocs(
  modules = [NLPModels]
)

deploydocs(deps = Deps.pip("mkdocs", "python-markdown-math"),
  repo = "github.com/abelsiqueira/NLPModels.jl.git",
  julia = "release",
  latest = "docs"
)
