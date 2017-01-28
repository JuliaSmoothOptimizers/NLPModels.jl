using Documenter, NLPModels

makedocs(
  modules = [NLPModels],
  doctest = false,
  assets = ["assets/style.css"],
  format = :html,
  sitename = "NLPModels.jl",
  pages = Any["Home" => "index.md",
              "Models" => "models.md",
              "Tutorial" => "tutorial.md",
              "API" => "api.md",
              "Reference" => "reference.md"]
)

deploydocs(deps = nothing, make = nothing,
  repo = "github.com/JuliaSmoothOptimizers/NLPModels.jl.git",
  target = "build",
  julia = "0.5",
  latest = "master"
)
