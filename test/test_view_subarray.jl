function test_view_subarray_nlp(nlp)
  @testset "Test view subarray of NLPs" begin
    n, m = nlp.meta.nvar, nlp.meta.ncon
    N = 2n
    Vidxs = [1:n, n.+(1:n), 1:2:N, collect(N:-2:1)]
    Cidxs = if m > 0
      N = 2m
      [1:m, m.+(1:m), 1:2:N, collect(N:-2:1)]
    else
      []
    end

    # Inputs
    x  = [-(-1.1)^i for i = 1:2n] # Instead of [1, -1, …], because it needs to
    v  = [-(-1.1)^i for i = 1:2n] # access different parts of the vector and
    y  = [-(-1.1)^i for i = 1:2m] # make a difference

    # Outputs
    g    = zeros(n)
    g2   = zeros(2n)
    c    = zeros(m)
    c2   = zeros(2m)
    jv   = zeros(m)
    jv2  = zeros(2m)
    jty  = zeros(n)
    jty2 = zeros(2n)
    hv   = zeros(n)
    hv2  = zeros(2n)

    for I = Vidxs
      xv = @view x[I]
      for foo in (obj, grad, hess)
        @test foo(nlp, x[I]) ≈ foo(nlp, xv)
      end

      # Some NLS don't implement hess_coord
      #rows1, cols1, vals1 = hess_coord(nlp, x[I])
      #rows2, cols2, vals2 = hess_coord(nlp, xv)
      #@test rows1 == rows2
      #@test cols1 == cols2
      #@test vals1 ≈ vals2

      if m > 0
        for foo in (cons, jac)
          @test foo(nlp, x[I]) ≈ foo(nlp, xv)
        end
        rows1, cols1, vals1 = jac_coord(nlp, x[I])
        rows2, cols2, vals2 = jac_coord(nlp, xv)
        @test rows1 == rows2
        @test cols1 == cols2
        @test vals1 ≈ vals2
      end

      for J = Cidxs
        yv = @view y[J]
        @test hess(nlp, x[I], y=y[J]) ≈ hess(nlp, xv, y=yv)
        yv = @view y[J]
        #rows1, cols1, vals1 = hess_coord(nlp, x[I], y=y[J])
        #rows2, cols2, vals2 = hess_coord(nlp, xv, y=yv)
        #@test rows1 == rows2
        #@test cols1 == cols2
        #@test vals1 ≈ vals2
      end

      # Inplace methods can have input and output as view, so 4 possibilities
      for J = Vidxs
        gv = @view g2[J]
        grad!(nlp, x[I],  g)
        grad!(nlp, x[I], gv); @test g ≈ g2[J]
        grad!(nlp,   xv, gv); @test g ≈ g2[J]
        grad!(nlp,   xv,  g); @test g ≈ g2[J]
      end

      for J = Cidxs
        cv = @view c2[J]
        cons!(nlp, x[I],  c)
        cons!(nlp, x[I], cv); @test c ≈ c2[J]
        cons!(nlp,   xv, cv); @test c ≈ c2[J]
        cons!(nlp,   xv,  c); @test c ≈ c2[J]
      end

      for J = Cidxs, K in Vidxs
        vv = @view v[K]
        jvv = @view jv2[J]
        @test jprod(nlp, x[I], v[K]) ≈ jprod(nlp, xv, vv)
        jprod!(nlp, x[I], v[K],  jv)
        jprod!(nlp, x[I], v[K], jvv); @test jv ≈ jv2[J]
        jprod!(nlp,   xv,   vv, jvv); @test jv ≈ jv2[J]
        jprod!(nlp,   xv,   vv,  jv); @test jv ≈ jv2[J]

        yv = @view y[J]
        jtyv = @view jty2[K]
        @test jtprod(nlp, x[I], y[J]) ≈ jtprod(nlp, xv, yv)
        jtprod!(nlp, x[I], y[J],  jty)
        jtprod!(nlp, x[I], y[J], jtyv); @test jty ≈ jty2[K]
        jtprod!(nlp,   xv,   yv, jtyv); @test jty ≈ jty2[K]
        jtprod!(nlp,   xv,   yv,  jty); @test jty ≈ jty2[K]
      end

      for J = Vidxs, K in Vidxs
        vv = @view v[J]
        hvv = @view hv2[K]
        @test hprod(nlp, x[I], v[J]) ≈ hprod(nlp, xv, vv)
        hprod!(nlp, x[I], v[J],  hv)
        hprod!(nlp, x[I], v[J], hvv); @test hv ≈ hv2[K]
        hprod!(nlp,   xv,   vv, hvv); @test hv ≈ hv2[K]
        hprod!(nlp,   xv,   vv,  hv); @test hv ≈ hv2[K]
        for P in Cidxs
          yv = @view y[P]
          @test hprod(nlp, x[I], v[J], y=y[P]) ≈ hprod(nlp, xv, vv, y=yv)
          hprod!(nlp, x[I], v[J],  hv, y=y[P])
          hprod!(nlp, x[I], v[J], hvv, y=y[P]); @test hv ≈ hv2[K]
          hprod!(nlp,   xv,   vv, hvv, y=yv);   @test hv ≈ hv2[K]
          hprod!(nlp,   xv,   vv,  hv, y=yv);   @test hv ≈ hv2[K]
        end
      end
    end
  end
end

function test_view_subarray_nls(nls)
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

function test_view_subarrays()
  @testset "Test view subarrays for many models" begin
    for p in problems
      nlp = eval(Symbol(p))()
      test_view_subarray_nlp(nlp)
    end
  end
end
