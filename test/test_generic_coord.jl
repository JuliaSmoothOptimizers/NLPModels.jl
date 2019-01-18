mutable struct TestGenericCoordNLPModel <: AbstractNLPModel
  meta :: NLPModelMeta
end

function TestGenericCoordNLPModel()
  TestGenericCoordNLPModel(NLPModelMeta(2, ncon=2))
end

#=
f(x) = x₁⁴ + 10x₂³
c(x) = [x₁ * x₂ - 2; x₁² + x₂² - 5]
=#

mutable struct TestGenericCoordNLSModel <: AbstractNLSModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  counters :: NLSCounters
end

function TestGenericCoordNLSModel()
  TestGenericCoordNLSModel(NLPModelMeta(2, ncon=2), NLSMeta(2, 2), NLSCounters())
end

#=
F(x) = [x₁ - 1; 10(x₂ - x₁²)]
c(x) = [x₁ * x₂ - 2; x₁² + x₂² - 5]
=#

function NLPModels.hess(nlp :: TestGenericCoordNLPModel, x :: AbstractVector;
              obj_weight :: Real=1.0, y :: AbstractVector=Float64[])
  return obj_weight * [12 * x[1]^2  0.0; 0.0  60 * x[2]] +
               y[1] * [        0.0  0.0; 1.0        0.0] +
               y[2] * [        2.0  0.0; 0.0        2.0]
end

function NLPModels.jac(nlp :: Union{TestGenericCoordNLPModel, TestGenericCoordNLSModel}, x :: AbstractVector)
  return [x[2] x[1]; 2x[1] 2x[2]]
end

function NLPModels.residual(nlp :: TestGenericCoordNLSModel, x :: AbstractVector)
  return [x[1] - 1; 10 * (x[2] - x[1]^2)]
end

function NLPModels.jac_residual(nlp :: TestGenericCoordNLSModel, x :: AbstractVector)
  return [1.0 0.0; -20x[1] 10.0]
end

function NLPModels.hess_residual(nlp :: TestGenericCoordNLSModel, x :: AbstractVector, i :: Int)
  if i == 1
    return zeros(2, 2)
  else
    return [-20.0 0; 0 0]
  end
end


function test_generic_coord()
  @testset "Testing generic jac_coord and hess_coord" begin
    gnlp = TestGenericCoordNLPModel()
    gnls = TestGenericCoordNLSModel()
    x = [1.0; 2.0]
    y = [3.0; 4.0]
    for nlp in [gnlp, gnls]
      @test hess(nlp, x, y=y) == sparse(hess_coord(nlp, x, y=y)..., 2, 2)
      @test jac(nlp, x) == sparse(jac_coord(nlp, x)..., 2, 2)
    end
  end
end

test_generic_coord()
