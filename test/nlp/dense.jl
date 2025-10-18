using Test
using NLPModels

@testset "ManualDenseNLPModel dense API" begin
  model = ManualDenseNLPModel()
  x = [0.5, 0.5]
  
  # Test objective and gradient
  @test obj(model, x) ≈ 0.5
  g = similar(x)
  grad!(model, x, g)
  @test g ≈ [1.0, 1.0]
  
  # Test constraints
  c = zeros(2)
  cons!(model, x, c)
  @test c[1] ≈ 0.0  # x₁ + x₂ - 1 = 0.5 + 0.5 - 1 = 0
  @test c[2] ≈ -1.5  # x₁² + x₂² - 2 = 0.5 - 2 = -1.5
  
  # Test Jacobian structure
  rows, cols = jac_structure(model)
  @test length(rows) == 4
  @test length(cols) == 4
  @test rows == [1, 1, 2, 2]
  @test cols == [1, 2, 1, 2]
  
  # Test Jacobian values
  vals = zeros(4)
  jac_coord!(model, x, vals)
  @test vals[1] ≈ 1.0   # ∂c₁/∂x₁
  @test vals[2] ≈ 1.0   # ∂c₁/∂x₂
  @test vals[3] ≈ 1.0   # ∂c₂/∂x₁ = 2*0.5
  @test vals[4] ≈ 1.0   # ∂c₂/∂x₂ = 2*0.5
  
  # Test Hessian structure
  rows_h, cols_h = hess_structure(model)
  @test length(rows_h) == 3
  @test length(cols_h) == 3
  @test rows_h == [1, 2, 2]
  @test cols_h == [1, 1, 2]
  
  # Test Hessian values
  y = [1.0, 1.0]
  vals_h = zeros(3)
  hess_coord!(model, x, y, vals_h)
  @test vals_h[1] ≈ 4.0  # ∇²L₁₁ = 2 (obj) + 2 (y₂)
  @test vals_h[2] ≈ 0.0  # ∇²L₂₁ = 0
  @test vals_h[3] ≈ 4.0  # ∇²L₂₂ = 2 (obj) + 2 (y₂)
  
  # Test that model is correctly typed as AbstractDenseNLPModel
  @test model isa AbstractDenseNLPModel
  @test model isa AbstractNLPModel
end
