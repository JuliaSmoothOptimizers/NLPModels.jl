@testset "A problem with zero variables doesn't make sense." begin
  @test_throws ErrorException NLPModelMeta(0)
end