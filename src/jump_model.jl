using JuMP

"Construct a `MathProgNLPModel` from a JuMP `Model`."
function MathProgNLPModel(jmodel :: JuMP.Model; kwargs...)
  JuMP.setsolver(jmodel, ModelReader())
  jmodel.internalModelLoaded || JuMP.build(jmodel)
  return MathProgNLPModel(jmodel.internalModel; kwargs...)
end

"Construct a `MathProgNLSModel` from a JuMP `Model` and a vector of
NLexpression.
"
function MathProgNLSModel(cmodel :: JuMP.Model,
                          F :: Vector{JuMP.NonlinearExpression}; kwargs...)
  # Hessian-Vector products don't work:
  # https://github.com/JuliaOpt/JuMP.jl/issues/1204
  @NLobjective(cmodel, Min, 0.5 * sum(Fi^2 for Fi in F))
  JuMP.setsolver(cmodel, ModelReader())
  cmodel.internalModelLoaded || JuMP.build(cmodel)
  ev = cmodel.internalModel.eval

  Fmodel = JuMP.Model()
  @NLobjective(Fmodel, Min, 0.0)
  Fmodel.nlpdata.user_operators = cmodel.nlpdata.user_operators
  @variable(Fmodel, x[1:MathProgBase.numvar(cmodel)])
  for Fi in F
    expr = ev.subexpressions_as_julia_expressions[Fi.index]
    replace!(expr, x)
    expr = :($expr == 0)
    JuMP.addNLconstraint(Fmodel, expr)
  end
  JuMP.setsolver(Fmodel, ModelReader())
  Fmodel.internalModelLoaded || JuMP.build(Fmodel)

  return MathProgNLSModel(Fmodel.internalModel,
                          cmodel.internalModel; kwargs...)
end

function replace!(ex, x)
  if isa(ex, Expr)
    for (i, arg) in enumerate(ex.args)
      if isa(arg, Expr)
        if arg.head == :ref && arg.args[1] == :x
          ex.args[i] = x[arg.args[2]]
        else
          replace!(arg, x)
        end
      end
    end
  end
end
