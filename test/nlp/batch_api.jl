@testset "Batch API" begin
    bnlp = BatchSimpleNLPModel([1.0, 2.0, 3.0])
    models = [
        SimpleNLPModel(; p = 1.0)
        SimpleNLPModel(; p = 2.0)
        SimpleNLPModel(; p = 3.0)
    ]

    @test bnlp.meta.nbatch == 3

    bx = [
        1.0 3.0 5.0;
        2.0 4.0 6.0;
    ]
    by = [
        -1.0 -3.0 -5.0;
        -2.0 -4.0 -6.0;
    ]
    xs = [
        1.0 3.0 5.0;
        2.0 4.0 6.0;
    ]
    ys = [
        -1.0 -3.0 -5.0;
        -2.0 -4.0 -6.0;
    ]
    bobj_weight = [1.0, 1.0, 1.0]

    bf = obj(bnlp, bx)
    bg = grad(bnlp, bx)
    bc = cons(bnlp, bx)
    bjvals = jac_coord(bnlp, bx)
    bhvals = hess_coord(bnlp, bx, by, bobj_weight)
    bJv = jprod(bnlp, bx, bx)
    bJtv = jtprod(bnlp, bx, by)
    bHv = hprod(bnlp, bx, by, bx, bobj_weight)
    jrows, jcols = jac_structure(bnlp)
    hrows, hcols = hess_structure(bnlp)

    for i in 1:3
        @test bf[i] == obj(models[i], xs[:,i])
        @test bg[:,i] == grad(models[i], xs[:,i])
        @test bc[:,i] == cons(models[i], xs[:,i])
        @test bjvals[:,i] == jac_coord(models[i], xs[:,i])
        @test bhvals[:,i] == hess_coord(models[i], xs[:,i], ys[:,i])
        @test bJv[:,i] == jprod(models[i], xs[:,i], xs[:,i])
        @test bJtv[:,i] == jtprod(models[i], xs[:,i], ys[:,i])
        @test bHv[:,i] == hprod(models[i], xs[:,i], ys[:,i], xs[:,i])
        jrowsi, jcolsi = jac_structure(models[i])
        @test jrows == jrowsi
        @test jcols == jcolsi
        hrowsi, hcolsi = hess_structure(models[i])
        @test hrows == hrowsi
        @test hcols == hcolsi
    end
end
