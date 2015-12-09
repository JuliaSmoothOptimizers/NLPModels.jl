# NLPModels

[![Build Status](https://travis-ci.org/JuliaOptimizers/NLPModels.jl.svg?branch=master)](https://travis-ci.org/JuliaOptimizers/NLPModels.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/l1rs9ajxkyc0cer9?svg=true)](https://ci.appveyor.com/project/dpo/nlpmodels-jl)
[![Coverage Status](https://coveralls.io/repos/JuliaOptimizers/NLPModels.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaOptimizers/NLPModels.jl?branch=master)

This package provides an interface to evaluate functions and derivatives in
optimization problems modeled using the
[`JuMP`](https://github.com/JuliaOpt/JuMP.jl) modeling language. The
interface is consistent with those of
[`AmplNLReader.jl`](https://github.com/dpo/AmplNLReader.jl) and
[`CUTEst.jl`](https://github.com/optimizers/CUTEst.jl).
