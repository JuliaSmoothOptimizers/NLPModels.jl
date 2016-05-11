# NLPModels

**OSX/Linux:**
master
[![Master Build Status](https://travis-ci.org/JuliaOptimizers/NLPModels.jl.svg?branch=master)](https://travis-ci.org/JuliaOptimizers/NLPModels.jl)
develop
[![Develop Build Status](https://travis-ci.org/JuliaOptimizers/NLPModels.jl.svg?branch=develop)](https://travis-ci.org/JuliaOptimizers/NLPModels.jl)

**Windows:**
master
[![Master Build status](https://ci.appveyor.com/api/projects/status/l1rs9ajxkyc0cer9/branch/master?svg=true)](https://ci.appveyor.com/project/dpo/nlpmodels-jl/branch/master)
develop
[![Develop Build status](https://ci.appveyor.com/api/projects/status/l1rs9ajxkyc0cer9/branch/develop?svg=true)](https://ci.appveyor.com/project/dpo/nlpmodels-jl/branch/develop)

**Coverage:**
master
[![Master Coverage Status](https://coveralls.io/repos/JuliaOptimizers/NLPModels.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaOptimizers/NLPModels.jl?branch=master)
develop
[![Develop Coverage Status](https://coveralls.io/repos/JuliaOptimizers/NLPModels.jl/badge.svg?branch=develop&service=github)](https://coveralls.io/github/JuliaOptimizers/NLPModels.jl?branch=develop)

This package provides an interface to evaluate functions and derivatives in optimization problems.
Problems may be modeled using the [`JuMP`](https://github.com/JuliaOpt/JuMP.jl) modeling language or may originate from an external modeling language, such as [AMPL](http://www.ampl.com) or [CUTEst](https://ccpforge.cse.rl.ac.uk/gf/project/cutest/wiki).
Corresponding interfaces are defined in [`AmplNLReader.jl`](https://github.com/JuliaOptimizers/AmplNLReader.jl) and [`CUTEst.jl`](https://github.com/JuliaOptimizers/CUTEst.jl).
