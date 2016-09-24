import JuMP

"Construct a `MathProgNLPModel` from a JuMP `Model`."
function MathProgNLPModel(jmodel :: JuMP.Model; kwargs...)
  JuMP.EnableNLPResolve()
  JuMP.setsolver(jmodel, ModelReader())
  jmodel.internalModelLoaded || JuMP.build(jmodel)
  return MathProgNLPModel(jmodel.internalModel; kwargs...)
end
