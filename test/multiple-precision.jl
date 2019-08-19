function multiple_precision(nlp :: AbstractNLPModel)
  @testset "Test multiple precision models" begin
    for T = [Float16, Float32, Float64, BigFloat]
      x = nlp.meta.x0
      @test typeof(obj(nlp, x)) == T
      @test eltype(grad(nlp, x)) == T
      @test eltype(hess(nlp, x)) == T
      @test eltype(hess(nlp, x, y=ones(T, 2))) == T
      @test eltype(hess(nlp, x, obj_weight=one(T), y=ones(T, 2))) == T
      @test eltype(cons(nlp, x)) == T
      @test eltype(jac(nlp, x)) == T
    end
  end
end

function multiple_precision(nls :: AbstractNLSModel)
  @testset "Test multiple precision models" begin
    for T = [Float16, Float32, Float64, BigFloat]
      x = nlp.meta.x0
      @test typeof(obj(nls, x)) == T
      @test eltype(grad(nls, x)) == T
      @test eltype(hess(nls, x)) == T
      @test eltype(hess(nlp, x, y=ones(T, 2))) == T
      @test eltype(hess(nlp, x, obj_weight=one(T), y=ones(T, 2))) == T
      @test eltype(cons(nls, x)) == T
      @test eltype(jac(nls, x)) == T
      @test eltype(residual(nls, x)) == T
      @test eltype(jac_residual(nls, x)) == T
      @test eltype(hess_residual(nls, x, ones(T, 3))) == T
    end
  end
end
