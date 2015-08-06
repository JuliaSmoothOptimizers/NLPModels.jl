# NLPModels

[![Build Status](https://travis-ci.org/dpo/NLPModels.jl.svg?branch=master)](https://travis-ci.org/dpo/NLPModels.jl)
[![Build
status](https://ci.appveyor.com/api/projects/status/l1rs9ajxkyc0cer9/branch/master?svg=true)](https://ci.appveyor.com/project/dpo/nlpmodels-jl/branch/master)
[![Coverage
Status](https://coveralls.io/repos/dpo/NLPModels.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/dpo/NLPModels.jl?branch=master)

This package provides an interface to evaluate functions and derivatives in
optimization problems modeled using the
[`JuMP`](https://github.com/JuliaOpt/JuMP.jl) modeling language. The
interface is consistent with those of
[`AmplNLReader.jl`](https://github.com/dpo/AmplNLReader.jl) and
[`CUTEst.jl`](https://github.com/optimizers/CUTEst.jl).
