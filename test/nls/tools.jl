mutable struct DummyNLSModel{T, S} <: AbstractNLSModel{T, S}
  nls_meta::NLSMeta{T, S}
end

@testset "Problem type functions" begin
  foo_list = [:nequ, :nvar, :x0, :nnzj, :nnzh, :nln, :nnln, :lin, :nlin]
  meta_list = [
    NLSMeta(2, 2),
    NLSMeta(2, 2, x0 = zeros(2)),
    NLSMeta(2, 3, lin = [1; 3]),
    NLSMeta(2, 3, lin = [2]),
    NLSMeta(1, 4, nnzh = 3, nnzj = 2),
  ]
  for f in fieldnames(NLSMeta), (j, meta) in enumerate(meta_list)
    nls = DummyNLSModel(meta)
    @test eval(Meta.parse("get_" * string(f)))(meta) == getproperty(meta, f)
    @test eval(Meta.parse("get_" * string(f)))(nls) == getproperty(meta, f)
  end
end
