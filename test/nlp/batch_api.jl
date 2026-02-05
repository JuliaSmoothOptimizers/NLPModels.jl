@testset "Batch API" begin
    bnlp = BatchSimpleNLPModel([1.0, 2.0, 3.0])
    models = [
        SimpleNLPModel(; p = 1.0)
        SimpleNLPModel(; p = 2.0)
        SimpleNLPModel(; p = 3.0)
    ]

    @test bnlp.meta.nbatch == 3

    bx = [
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    ]
    by = [
        -1.0, -2.0,
        -3.0, -4.0,
        -5.0, -6.0,
    ]
    xs = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ]
    ys = [
        [-1.0, -2.0],
        [-3.0, -4.0],
        [-5.0, -6.0],
    ]
    bobj_weight = [1.0, 1.0, 1.0]

    bf = batch_obj(bnlp, bx)
    bg = batch_grad(bnlp, bx)
    bc = batch_cons(bnlp, bx)
    bjvals = batch_jac_coord(bnlp, bx)
    bhvals = batch_hess_coord(bnlp, bx, by, bobj_weight)
    bJv = batch_jprod(bnlp, bx, bx)
    bJtv = batch_jtprod(bnlp, bx, by)
    bHv = batch_hprod(bnlp, bx, by, bx, bobj_weight)
    jrows, jcols = batch_jac_structure(bnlp)
    hrows, hcols = batch_hess_structure(bnlp)

    for i in 1:3
        @test bf[i] == obj(models[i], xs[i])
        @test bg[(i-1)*2+1:(i)*2] == grad(models[i], xs[i])
        @test bc[(i-1)*2+1:(i)*2] == cons(models[i], xs[i])
        @test bjvals[(i-1)*4+1:(i)*4] == jac_coord(models[i], xs[i])
        @test bhvals[(i-1)*2+1:(i)*2] == hess_coord(models[i], xs[i], ys[i])
        @test bJv[(i-1)*2+1:(i)*2] == jprod(models[i], xs[i], xs[i])
        @test bJtv[(i-1)*2+1:(i)*2] == jtprod(models[i], xs[i], ys[i])
        @test bHv[(i-1)*2+1:(i)*2] == hprod(models[i], xs[i], ys[i], xs[i])
        jrowsi, jcolsi = jac_structure(models[i])
        @test jrows == jrowsi
        @test jcols == jcolsi
        hrowsi, hcolsi = hess_structure(models[i])
        @test hrows == hrowsi
        @test hcols == hcolsi
    end
end
