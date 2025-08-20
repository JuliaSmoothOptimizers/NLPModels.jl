using Test
using NLPModels

@testset "ManualDenseNLPModel dense API" begin
    jac = [1.0 2.0; 3.0 4.0]
    hess = [10.0 20.0; 20.0 40.0]
    model = ManualDenseNLPModel{Float64, Vector{Float64}}(jac, hess)
    x = [0.0, 0.0]
    @test jac_coord(model, x) == jac
    @test hess_coord(model, x) == hess
    rows, cols = jac_structure(model)
    @test length(rows) == 4 && length(cols) == 4
    rows_h, cols_h = hess_structure(model)
    @test length(rows_h) == 4 && length(cols_h) == 4
end
