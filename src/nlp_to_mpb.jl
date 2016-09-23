# Convert an AbstractNLPModel to a MathProgBase model that can be passed
# to a standard solver.

export NLPtoMPB


"""
    mp = NLPtoMPB(nlp, solver)

Return a `MathProgBase` model corresponding to an `AbstractNLPModel`.

#### Arguments
- `nlp::AbstractNLPModel`
- `solver::AbstractMathProgSolver` a solver instance, e.g., `IpoptSolver()`

Currently, all models are treated as nonlinear models.

#### Return values
The function returns a `MathProgBase` model `mpbmodel` such that it should
be possible to call

    MathProgBase.optimize!(mpbmodel)
"""
NLPtoMPB(model :: AbstractNLPModel, args...) = throw(NotImplementedError("NLPtoMPB"))

function NLPtoMPB(model :: MathProgNLPModel, solver :: MathProgBase.AbstractMathProgSolver)
  mpbmodel = MathProgBase.NonlinearModel(solver)
  MathProgBase.loadproblem!(mpbmodel,
                            model.meta.nvar, model.meta.ncon,
                            model.meta.lvar, model.meta.uvar,
                            model.meta.lcon, model.meta.ucon,
                            :Min, model.mpmodel.eval)
  MathProgBase.setwarmstart!(mpbmodel, model.meta.x0)
  return mpbmodel
end
