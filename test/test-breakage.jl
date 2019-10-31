using Git, GitHub, Pkg

"""
    test_breakage()

Downloads each package and check if tests pass using the master of NLPModels.jl.
Packages are checked in two situations:

- Tagged version
- master branch

If any tagged version breaks then the next version should be a major update or
the broken package has bugs.
"""
function test_breakage()
  packages = ["AmplNLReader", "CaNNOLeS", "CUTEst", "QuadraticModels", "NLPModelsIpopt",
              "NLPModelsJuMP", "SolverTools"]

  msg(s) = printstyled(s, color=:red, bold=true)
  msgn(s) = printstyled(s * "\n", color=:red, bold=true)

  passing = fill(false, length(packages))
  passing_master = fill(false, length(packages))
  tagged = fill("", length(packages))

  master_fail = "![](https://img.shields.io/badge/master-Fail-red)"
  master_pass = "![](https://img.shields.io/badge/master-Pass-green)"
  version_fail(n) = "![](https://img.shields.io/badge/$n-Fail-red)"
  version_pass(n) = "![](https://img.shields.io/badge/$n-Pass-green)"

  thispath = pwd()
  mkpath("test-breakage")
  try
    for (i,package) in enumerate(packages)
      msgn("#"^100)
      msgn("  Testing package $package")
      msgn("#"^100)
      cd(joinpath(thispath, "test-breakage"))
      try
        url = "https://github.com/JuliaSmoothOptimizers/$package.jl"
        Git.run(`clone $url`)
        cd("$package.jl")
        pkg"activate ."
        pkg"instantiate"
        pkg"dev ../.."
        pkg"build"
        pkg"test"
        passing_master[i] = true
      catch e
        println(e)
      end

      cd(joinpath(thispath, "test-breakage"))
      try
        # Testing on last tagged version
        cd("$package.jl")
        tag = split(Git.readstring(`tag`))[end]
        tagged[i] = tag
        Git.run(`checkout $tag`)
        pkg"up"
        pkg"build"
        pkg"test"
        passing[i] = true
      catch e
        println(e)
      end
    end
  catch ex
    @error ex
  finally
    cd(thispath)
  end

  msgn("Summary\n")
  for (i,package) in enumerate(packages)
    msgn("- $package")
    msgn("  Master branch: " * (passing_master[i] ? "✓" : "x"))
    msgn("  Version $(tagged[i]): " * (passing[i] ? "✓" : "x"))
  end

  if lowercase(get(ENV, "TRAVIS", "false")) == "true"
    if get(ENV, "GITHUB_AUTH", nothing) === nothing
      @warn "GITHUB_AUTH not found, skipping comment push"
      return
    end
    myauth = GitHub.authenticate(ENV["GITHUB_AUTH"])
    myrepo = repo(ENV["TRAVIS_PULL_REQUEST_SLUG"], auth=myauth) # "JuliaSmoothOptimizers/NLPModels.jl"
    prs = pull_requests(myrepo, auth=myauth)
    pr = nothing
    for p in prs[1]
      if p.merge_commit_sha == ENV["TRAVIS_COMMIT"]
        pr = p
      end
    end

    output = ":robot: Testing breakage of this commit\n\n"
    output *= "| Package Name | master | Tagged |\n"
    output *= "|--|--|--|\n"
    for (i,package) in enumerate(packages)
      output *= "| $package | "
      output *= (passing_master[i] ? master_pass : master_fail) * " | "
      output *= (passing[i] ? version_pass(tagged[i]) : version_fail(tagged[i])) * " |\n"
    end
    create_comment(myrepo, pr, output, auth=myauth)
  end
end

test_breakage()
