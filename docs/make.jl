using Documenter, NLPModels

makedocs(
  modules = [NLPModels],
  doctest = true,
  linkcheck = true,
  strict = true,
  format = Documenter.HTML(assets = ["assets/style.css"], prettyurls = get(ENV, "CI", nothing) == "true"),
  sitename = "NLPModels.jl",
  pages = ["Home" => "index.md",
           "Models" => "models.md",
           "Tools" => "tools.md",
           "Tutorial" => "tutorial.md",
           "API" => "api.md",
           "Reference" => "reference.md"
          ]
)

deploydocs(repo = "github.com/JuliaSmoothOptimizers/NLPModels.jl.git")
