@testset "Parametric API" begin
  #=
  Problem:
    min   f(x, p) = (x₁ - p₁)² + (x₂ - p₂)²
    s.t.  lcon(p) = [-p₂] ≤ c(x, p) = x₁ + p₁*x₂ ≤ ucon(p) = [p₂]
          lvar(p) = [-p₁; -p₂] ≤ x ≤ uvar(p) = [p₁; p₂]

  Parametric derivatives:
    ∇ₚf(x)         = [-2(x₁-p₁); -2(x₂-p₂)]
    Jₚ(x)          = [[x₂, 0]]
    Hₚ(x,y)        = [-2σ  0; y₁  -2σ]
    ∂lcon/∂p        = [[0, -1]]
    ∂ucon/∂p        = [[0,  1]]
    ∂lvar/∂p        = -I₂
    ∂uvar/∂p        =  I₂
  =#

  nlp = SimpleParamNLPModel()
  @test eltype(nlp) == Float64

  n = nlp.meta.nvar
  m = nlp.meta.ncon
  np = nlp.meta.nparam
  p1, p2 = nlp.ps

  @test np == 2
  @test nlp.meta.nnzjp    == 1
  @test nlp.meta.nnzhp    == 3
  @test nlp.meta.nnzgp    == 2
  @test nlp.meta.nnzjplcon == 1
  @test nlp.meta.nnzjpucon == 1
  @test nlp.meta.nnzjplvar == 2
  @test nlp.meta.nnzjpuvar == 2

  x = [1.0, 2.0]
  y = [0.5]
  v_p = [0.3, 0.7]
  v_x = [0.4, 0.6]
  v_c = [1.2]
  σ = 2.0

  @testset "grad_param" begin
    gp_exact = [-2 * (x[1] - p1); -2 * (x[2] - p2)]
    @test grad_param(nlp, x) ≈ gp_exact
    gp = zeros(np)
    @test grad_param!(nlp, x, gp) ≈ gp_exact
    @test gp ≈ gp_exact
  end

  @testset "jac_param" begin
    rows, cols = jac_param_structure(nlp)
    @test rows == [1]
    @test cols == [1]

    vals_exact = [x[2]]
    vals = jac_param_coord(nlp, x)
    @test vals ≈ vals_exact
    vals2 = zeros(nlp.meta.nnzjp)
    @test jac_param_coord!(nlp, x, vals2) ≈ vals_exact
    @test vals2 ≈ vals_exact
  end

  @testset "jpprod / jptprod" begin
    Jv_exact = [x[2] * v_p[1]]
    @test jpprod(nlp, x, v_p) ≈ Jv_exact
    Jv = zeros(m)
    @test jpprod!(nlp, x, v_p, Jv) ≈ Jv_exact
    @test Jv ≈ Jv_exact

    Jtv_exact = [x[2] * v_c[1]; 0.0]
    @test jptprod(nlp, x, v_c) ≈ Jtv_exact
    Jtv = zeros(np)
    @test jptprod!(nlp, x, v_c, Jtv) ≈ Jtv_exact
    @test Jtv ≈ Jtv_exact
  end

  @testset "hess_param" begin
    rows, cols = hess_param_structure(nlp)
    @test rows == [1, 2, 2]
    @test cols == [1, 1, 2]

    # objective only (y=0, σ=1)
    vals_obj = hess_param_coord(nlp, x)
    @test vals_obj ≈ [-2.0, 0.0, -2.0]

    # Lagrangian (y=[0.5], σ=1)
    vals_lag = hess_param_coord(nlp, x, y)
    @test vals_lag ≈ [-2.0, y[1], -2.0]

    # with obj_weight
    vals_w = hess_param_coord(nlp, x, y; obj_weight = σ)
    @test vals_w ≈ [-2σ, y[1], -2σ]

    vals2 = zeros(nlp.meta.nnzhp)
    @test hess_param_coord!(nlp, x, y, vals2; obj_weight = σ) ≈ [-2σ, y[1], -2σ]
    @test vals2 ≈ [-2σ, y[1], -2σ]
  end

  @testset "hpprod / hptprod" begin
    # Hₚ = [-2σ  0; y₁  -2σ],  Hₚ*v_p = [-2σ*v_p[1]; y₁*v_p[1] - 2σ*v_p[2]]
    Hv_exact = [-2σ * v_p[1]; y[1] * v_p[1] - 2σ * v_p[2]]
    @test hpprod(nlp, x, y, v_p; obj_weight = σ) ≈ Hv_exact
    Hv = zeros(n)
    @test hpprod!(nlp, x, y, v_p, Hv; obj_weight = σ) ≈ Hv_exact
    @test Hv ≈ Hv_exact

    # objective only (y=0, σ=1): Hₚ = [-2  0; 0  -2]
    @test hpprod(nlp, x, v_p) ≈ [-2 * v_p[1]; -2 * v_p[2]]

    # Hₚᵀ*v_x = [-2σ*v_x[1] + y₁*v_x[2]; -2σ*v_x[2]]
    Htv_exact = [-2σ * v_x[1] + y[1] * v_x[2]; -2σ * v_x[2]]
    @test hptprod(nlp, x, y, v_x; obj_weight = σ) ≈ Htv_exact
    Htv = zeros(np)
    @test hptprod!(nlp, x, y, v_x, Htv; obj_weight = σ) ≈ Htv_exact
    @test Htv ≈ Htv_exact
  end

  @testset "lcon_jac" begin
    rows, cols = lcon_jac_param_structure(nlp)
    @test rows == [1]
    @test cols == [2]

    vals = lcon_jac_param_coord(nlp)
    @test vals ≈ [-1.0]
    vals2 = zeros(nlp.meta.nnzjplcon)
    @test lcon_jac_param_coord!(nlp, vals2) ≈ [-1.0]

    # ∂lcon/∂p = [[0, -1]],  Jv = [0, -1] * v_p = -v_p[2]
    Jv_exact = [-v_p[2]]
    @test lcon_jpprod(nlp, v_p) ≈ Jv_exact
    Jv = zeros(m)
    @test lcon_jpprod!(nlp, v_p, Jv) ≈ Jv_exact

    # Jᵀv = [0; -1] * v_c = [0; -v_c[1]]
    Jtv_exact = [0.0; -v_c[1]]
    @test lcon_jptprod(nlp, v_c) ≈ Jtv_exact
    Jtv = zeros(np)
    @test lcon_jptprod!(nlp, v_c, Jtv) ≈ Jtv_exact
  end

  @testset "ucon_jac" begin
    rows, cols = ucon_jac_param_structure(nlp)
    @test rows == [1]
    @test cols == [2]

    vals = ucon_jac_param_coord(nlp)
    @test vals ≈ [1.0]
    vals2 = zeros(nlp.meta.nnzjpucon)
    @test ucon_jac_param_coord!(nlp, vals2) ≈ [1.0]

    # ∂ucon/∂p = [[0, 1]],  Jv = v_p[2]
    Jv_exact = [v_p[2]]
    @test ucon_jpprod(nlp, v_p) ≈ Jv_exact
    Jv = zeros(m)
    @test ucon_jpprod!(nlp, v_p, Jv) ≈ Jv_exact

    # Jᵀv = [0; 1] * v_c = [0; v_c[1]]
    Jtv_exact = [0.0; v_c[1]]
    @test ucon_jptprod(nlp, v_c) ≈ Jtv_exact
    Jtv = zeros(np)
    @test ucon_jptprod!(nlp, v_c, Jtv) ≈ Jtv_exact
  end

  @testset "lvar_jac" begin
    rows, cols = lvar_jac_param_structure(nlp)
    @test rows == [1, 2]
    @test cols == [1, 2]

    vals = lvar_jac_param_coord(nlp)
    @test vals ≈ [-1.0, -1.0]
    vals2 = zeros(nlp.meta.nnzjplvar)
    @test lvar_jac_param_coord!(nlp, vals2) ≈ [-1.0, -1.0]

    # ∂lvar/∂p = -I₂,  Jv = -v_p
    @test lvar_jpprod(nlp, v_p) ≈ -v_p
    Jv = zeros(n)
    @test lvar_jpprod!(nlp, v_p, Jv) ≈ -v_p

    # Jᵀv = -v_x
    @test lvar_jptprod(nlp, v_x) ≈ -v_x
    Jtv = zeros(np)
    @test lvar_jptprod!(nlp, v_x, Jtv) ≈ -v_x
  end

  @testset "uvar_jac" begin
    rows, cols = uvar_jac_param_structure(nlp)
    @test rows == [1, 2]
    @test cols == [1, 2]

    vals = uvar_jac_param_coord(nlp)
    @test vals ≈ [1.0, 1.0]
    vals2 = zeros(nlp.meta.nnzjpuvar)
    @test uvar_jac_param_coord!(nlp, vals2) ≈ [1.0, 1.0]

    # ∂uvar/∂p = I₂,  Jv = v_p
    @test uvar_jpprod(nlp, v_p) ≈ v_p
    Jv = zeros(n)
    @test uvar_jpprod!(nlp, v_p, Jv) ≈ v_p

    # Jᵀv = v_x
    @test uvar_jptprod(nlp, v_x) ≈ v_x
    Jtv = zeros(np)
    @test uvar_jptprod!(nlp, v_x, Jtv) ≈ v_x
  end

  @testset "non-parametric behavior" begin
    plain = SimpleNLPModel()
    @test plain.meta.nparam    == 0
    @test plain.meta.nnzjp     == 0
    @test plain.meta.nnzhp     == 0
    @test plain.meta.nnzgp     == 0
    @test plain.meta.nnzjplcon == 0
    @test plain.meta.nnzjpucon == 0
    @test plain.meta.nnzjplvar == 0
    @test plain.meta.nnzjpuvar == 0
    @test !plain.meta.grad_param_available
    @test !plain.meta.jac_param_available
    @test !plain.meta.hess_param_available
  end
end
