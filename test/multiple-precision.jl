function multiple_precision()
  @testset "Test multiple precision models" begin
    for T = [Float16, Float32, Float64, BigFloat]
      nlp = ADNLPModel(x->sum(x.^4), ones(T, 2),
                       c=x->[x[1]^2 + x[2]^2 - 1; x[1] * x[2]],
                       lcon=zeros(T, 2), ucon=zeros(T, 2))
      x = nlp.meta.x0
      @test typeof(obj(nlp, x)) == T
      @test eltype(grad(nlp, x)) == T
      @test eltype(hess(nlp, x)) == T
      @test eltype(hess(nlp, x, y=ones(T, 2))) == T
      @test eltype(hess(nlp, x, obj_weight=one(T), y=ones(T, 2))) == T
      @test eltype(cons(nlp, x)) == T
      @test eltype(jac(nlp, x)) == T

      nls = ADNLSModel(x->[x[1] - 1; exp(x[2]) - x[1]; sin(x[1]) * x[2]], ones(T, 2), 3,
                       c=x->[x[1]^2 + x[2]^2 - 1; x[1] * x[2]],
                       lcon=zeros(T, 2), ucon=zeros(T, 2))
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

multiple_precision()
