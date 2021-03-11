export view_subarray_nls

"""
    view_subarray_nls(nls)

Check that the API work with views, and that the results is correct.
"""
function view_subarray_nls(nls)
  @testset "Test view subarray of NLSs" begin
    n, ne = nls.meta.nvar, nls.nls_meta.nequ
    N = 2n
    Vidxs = [1:n, n.+(1:n), 1:2:N, collect(N:-2:1)]
    N = 2ne
    Fidxs = [1:ne, ne.+(1:ne), 1:2:N, collect(N:-2:1)]

    # Inputs
    x  = [-(-1.1)^i for i = 1:2n] # Instead of [1, -1, …], because it needs to
    v  = [-(-1.1)^i for i = 1:2n] # access different parts of the vector and
    y  = [-(-1.1)^i for i = 1:2ne] # make a difference

    # Outputs
    F    = zeros(ne)
    F2   = zeros(2ne)
    jv   = zeros(ne)
    jv2  = zeros(2ne)
    jty  = zeros(n)
    jty2 = zeros(2n)
    hv   = zeros(n)
    hv2  = zeros(2n)

    for I = Vidxs
      xv = @view x[I]
      for foo in (residual, jac_residual)
        @test foo(nls, x[I]) ≈ foo(nls, xv)
      end

      # Inplace methods can have input and output as view, so 4 possibilities
      for J = Fidxs
        Fv = @view F2[J]
        residual!(nls, x[I],  F)
        residual!(nls, x[I], Fv); @test F ≈ F2[J]
        residual!(nls,   xv, Fv); @test F ≈ F2[J]
        residual!(nls,   xv,  F); @test F ≈ F2[J]
      end

      for J = Fidxs, K in Vidxs
        vv = @view v[K]
        jvv = @view jv2[J]
        @test jprod_residual(nls, x[I], v[K]) ≈ jprod_residual(nls, xv, vv)
        jprod_residual!(nls, x[I], v[K],  jv)
        jprod_residual!(nls, x[I], v[K], jvv); @test jv ≈ jv2[J]
        jprod_residual!(nls,   xv,   vv, jvv); @test jv ≈ jv2[J]
        jprod_residual!(nls,   xv,   vv,  jv); @test jv ≈ jv2[J]

        yv = @view y[J]
        jtyv = @view jty2[K]
        @test jtprod_residual(nls, x[I], y[J]) ≈ jtprod_residual(nls, xv, yv)
        jtprod_residual!(nls, x[I], y[J],  jty)
        jtprod_residual!(nls, x[I], y[J], jtyv); @test jty ≈ jty2[K]
        jtprod_residual!(nls,   xv,   yv, jtyv); @test jty ≈ jty2[K]
        jtprod_residual!(nls,   xv,   yv,  jty); @test jty ≈ jty2[K]
      end

      for i = 1:ne
        @test jth_hess_residual(nls, x[I], i) ≈ jth_hess_residual(nls, xv, i)

        for J = Vidxs, K in Vidxs
          vv = @view v[J]
          hvv = @view hv2[K]
          @test hprod_residual(nls, x[I], i, v[J]) ≈ hprod_residual(nls, xv, i, vv)
          hprod_residual!(nls, x[I], i, v[J],  hv)
          hprod_residual!(nls, x[I], i, v[J], hvv); @test hv ≈ hv2[K]
          hprod_residual!(nls,   xv, i,   vv, hvv); @test hv ≈ hv2[K]
          hprod_residual!(nls,   xv, i,   vv,  hv); @test hv ≈ hv2[K]
        end
      end
    end
  end
end