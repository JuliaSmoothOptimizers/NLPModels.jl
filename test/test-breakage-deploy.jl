using Git, GitHub, JSON, Pkg

"""
    test_breakage_deploy()

Read files from breakage-info and publish to the PR
"""
function test_breakage_deploy()
  if lowercase(get(ENV, "TRAVIS", "false")) == "false"
    error("Only run in travis")
  end
  if get(ENV, "GITHUB_AUTH", nothing) === nothing
    error("GITHUB_AUTH not found")
  end
  key = ENV["GITHUB_AUTH"]
  repo = ENV["TRAVIS_REPO_SLUG"]
  user = split(repo, "/")[1]
  upstream = "https://$user:$key@github.com/$repo"
  Git.run(`remote add upstream $upstream`)
  Git.run(`fetch upstream`)
  Git.run(`checkout -f breakage-info`)

  badge_pass(x) = "![](https://img.shields.io/badge/$x-Pass-green)"
  badge_fail(x) = "![](https://img.shields.io/badge/$x-Fail-red)"
  badge(tf, x) = tf ? badge_pass(x) : badge_fail(x)

  packages = ["AmplNLReader", "CUTEst", "CaNNOLeS", "NLPModelsIpopt", "NLPModelsJuMP", "QuadraticModels", "SolverTools"]

  output = ":robot: Testing breakage of this pull request\n\n"
  output *= "| Package Name | master | stable |\n"
  output *= "|--|--|--|\n"
  for package in packages
    output *= "| $package | "

    for version in ["master", "stable"]
      info = JSON.parse(open("$package-$version"))
      bdg = badge(info["pass"], info["tag"])
      url = info["travis_link"]
      output *= "[$bdg]($url) | "
    end
    output *= "\n"
  end

  println(output)

  myauth = GitHub.authenticate(key)
  myrepo = GitHub.repo(ENV["TRAVIS_PULL_REQUEST_SLUG"], auth=myauth) # "JuliaSmoothOptimizers/NLPModels.jl"
  prs = pull_requests(myrepo, auth=myauth)
  pr = nothing
  for p in prs[1]
    if p.merge_commit_sha == ENV["TRAVIS_COMMIT"]
      pr = p
    end
  end
  @assert pr != nothing

  GitHub.create_comment(GitHub.DEFAULT_API, myrepo, pr, :pr, auth=myauth, params=Dict(:body => output))
end

test_breakage_deploy()
