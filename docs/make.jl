using Documenter, NLPModels

makedocs(
  modules = [NLPModels],
  doctest = true,
  linkcheck = true,
  format = Documenter.HTML(
    assets = ["assets/style.css"],
    prettyurls = get(ENV, "CI", nothing) == "true",
  ),
  sitename = "NLPModels.jl",
  pages = [
    "Home" => "index.md",
    "Models" => "models.md",
    "Guidelines" => "guidelines.md",
    "Tools" => "tools.md",
    "API" => "api.md",
    "Internals" => "internals.md",
    "Reference" => "reference.md",
  ],
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/NLPModels.jl.git",
  push_preview = true,
  devbranch = "main",
)
