module TestBoundsAnalysis

using Test
import NLPModels

# Model GPU storage: copying to the host is supported, scalar access is not.
struct HostConvertibleVector{T} <: AbstractVector{T}
  data::Vector{T}
end

Base.size(v::HostConvertibleVector) = size(v.data)
Base.getindex(::HostConvertibleVector, ::Int) = error("Scalar indexing is disallowed")
Base.convert(::Type{Vector{T}}, v::HostConvertibleVector{T}) where {T} = copy(v.data)

@testset "Bound analysis without scalar indexing" begin
  @testset "$T" for T in (Float32, Float64, BigFloat)
    lower = HostConvertibleVector(T[1, 0, -Inf, 0, -Inf, 2])
    upper = HostConvertibleVector(T[1, Inf, 1, 1, Inf, 1])
    x0 = HostConvertibleVector(zeros(T, 6))
    meta = NLPModels.NLPModelMeta{T, typeof(x0)}(
      6,
      x0 = x0,
      lvar = lower,
      uvar = upper,
      ncon = 6,
      y0 = HostConvertibleVector(zeros(T, 6)),
      lcon = lower,
      ucon = upper,
    )
    @test meta.lvar === lower
    @test meta.uvar === upper
    @test meta.lcon === lower
    @test meta.ucon === upper
    for (variable_set, constraint_set, index) in (
      (:ifix, :jfix, 1),
      (:ilow, :jlow, 2),
      (:iupp, :jupp, 3),
      (:irng, :jrng, 4),
      (:ifree, :jfree, 5),
      (:iinf, :jinf, 6),
    )
      @test getproperty(meta, variable_set) == [index]
      @test getproperty(meta, constraint_set) == [index]
    end
  end
end

end # module TestBoundsAnalysis
