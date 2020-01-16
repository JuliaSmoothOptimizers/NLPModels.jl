using Git, JSON, Pkg

"""
    test_breakage()

Downloads a package (given by ENV["PKG"]) and check if its tests pass using the pull request version of NLPModels.jl.
Packages are checked in two situations:

- stable version (ENV["stable"])
- master branch  (ENV["master"])

If any tagged version breaks then the next version should be a major update or
the broken package has bugs.
"""
function test_breakage()
  if lowercase(get(ENV, "TRAVIS", "false")) == "false"
    error("Only run in travis")
  end
  if get(ENV, "GITHUB_AUTH", nothing) === nothing
    error("GITHUB_AUTH not found")
  end
  key = ENV["GITHUB_AUTH"]

  package = get(ENV, "PKG", nothing)
  version = get(ENV, "VERSION", nothing)
  package === nothing || version === nothing && error("No environment variable PKG defined")
  tag = version

  passing = false
  thispath = pwd()
  mkpath("test-breakage")
  try
    println("#"^100)
    println("  Testing package $package")
    println("#"^100)

    cd(joinpath(thispath, "test-breakage"))
    url = "https://github.com/JuliaSmoothOptimizers/$package.jl"
    Git.run(`clone $url`)
    cd("$package.jl")
    if version == "stable"
      tag = split(Git.readstring(`tag`))[end]
      Git.run(`checkout $tag`)
    end
    pkg"activate ."
    pkg"instantiate"
    pkg"dev ../.."
    pkg"build"
    pkg"test"
    passing = true
  catch e
    println(e)
  end

  # Enter branch breakage-info and commit file $package-$version
  info = Dict(:pass => passing,
              :travis_link => ENV["TRAVIS_JOB_WEB_URL"],
              :pr => ENV["TRAVIS_PULL_REQUEST"],
              :tag => tag
             )
  cd(thispath)
  repo = ENV["TRAVIS_REPO_SLUG"]
  user = split(repo, "/")[1]
  upstream = "https://$user:$key@github.com/$repo"
  Git.run(`remote add upstream $upstream`)
  Git.run(`fetch upstream`)
  if !success(`git checkout -f -b breakage-info upstream/breakage-info`)
    Git.run(`checkout --orphan breakage-info`)
    Git.run(`reset --hard`)
    Git.run(`commit --allow-empty -m "Initial commit"`)
  end
  open("$package-$version", "w") do io
    JSON.print(io, info, 2)
  end
  Git.run(`add $package-$version`)
  Git.run(`commit -m ":robot: test-breakage of $package-$version"`)
  tries = 0
  while !Git.success(`push upstream breakage-info`)
    if tries > 10
      error("Too many failures in pushing")
    end
    @warn("push failed, trying again")
    sleep(5)
    Git.run(`fetch upstream`)
    Git.run(`merge upstream/breakage-info`)
    tries += 1
  end
end

test_breakage()
