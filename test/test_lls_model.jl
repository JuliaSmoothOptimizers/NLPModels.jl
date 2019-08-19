function lls_test()
  @testset "lls_test" begin
    for A = [Matrix(1.0I, 10, 3) .+ 1, sparse(1.0I, 10, 3) .+ 1],
        C = [ones(1, 3), [ones(1,3); -I], sparse(ones(1,3))]
      b = collect(1:10)
      nequ, nvar = size(A)
      ncon = size(C,1)
      nls = LLSModel(A, b, C=C, lcon=zeros(ncon), ucon=zeros(ncon))
      x = [1.0; -1.0; 1.0]

      @test isapprox(A * x - b, residual(nls, x), rtol=1e-8)
      @test A == jac_residual(nls, x)
      @test A == sparse(jac_coord_residual(nls, x)..., nequ, nvar)
      @test sparse(hess_coord_residual(nls, x, ones(nequ))..., nvar, nvar) == zeros(nvar, nvar)
      @test hess_residual(nls, x, ones(10)) == zeros(3,3)
      for i = 1:10
        @test isapprox(zeros(3, 3), jth_hess_residual(nls, x, i), rtol=1e-8)
      end

      I, J, V = jac_coord(nls, x)
      @test sparse(I, J, V, ncon, nvar) == C

      @test nls.meta.nlin == length(nls.meta.lin) == ncon
      @test nls.meta.nnln == length(nls.meta.nln) == 0
    end
  end

  @testset "Consistency of Linear problem" begin
    m, n = 50, 20
    A = Matrix(1.0I, m, n) .+ 1
    b = collect(1:m)
    lvar = -ones(n)
    uvar = ones(n)
    lls_model = LLSModel(A, b, lvar=lvar, uvar=uvar)
    nlss = [lls_model]

    consistent_nlss(nlss)
  end

  @testset "Consistency of Linear problem with linear constraints" begin
    m, n = 50, 20
    A = Matrix(1.0I, m, n) .+ 1
    b = collect(1:m)
    lvar = -ones(n)
    uvar = ones(n)
    nc = 10
    C = [ones(nc, n); 2 * ones(nc, n); -ones(nc, n); -Matrix(1.0I, nc, n)]
    lcon = [   zeros(nc); -ones(nc); fill(-Inf,nc); zeros(nc)]
    ucon = [fill(Inf,nc);  ones(nc);     zeros(nc); zeros(nc)]
    K = ((1:4:4nc) .+ (0:3)')[:]
    lcon, ucon = lcon[K], ucon[K]
    lls_model = LLSModel(A, b, lvar=lvar, uvar=uvar, C=C, lcon=lcon,
                         ucon=ucon)
    nlss = [lls_model]

    consistent_nlss(nlss)
  end

  @testset "Consistency of LLS with Matrix and LinearOperator" begin
    m, n = 50, 20
    A = Matrix(1.0I, m, n) .+ 1
    b = collect(1:m)
    lls = LLSModel(A, b)
    lls2 = LLSModel(LinearOperator(A), b)
    nlss = [lls, lls2]

    consistent_nlss(nlss, exclude=[jac_residual])

    nc = 10
    C = [ones(nc, n); 2 * ones(nc, n); -ones(nc, n); -Matrix(1.0I, nc, n)]
    lcon = [   zeros(nc); -ones(nc); fill(-Inf,nc); zeros(nc)]
    ucon = [fill(Inf,nc);  ones(nc);     zeros(nc); zeros(nc)]
    K = ((1:4:4nc) .+ (0:3)')[:]
    lcon, ucon = lcon[K], ucon[K]
    lls  = LLSModel(A, b, C=C, lcon=lcon, ucon=ucon)
    lls2 = LLSModel(LinearOperator(A), b, C=C, lcon=lcon, ucon=ucon)
    lls3 = LLSModel(A, b, C=LinearOperator(C), lcon=lcon, ucon=ucon)
    lls4 = LLSModel(LinearOperator(A), b, C=LinearOperator(C), lcon=lcon, ucon=ucon)
    nlss = [lls, lls2, lls3, lls4]

    consistent_nlss(nlss, exclude=[jac_residual, hess_coord_residual])
  end
end

lls_test()
