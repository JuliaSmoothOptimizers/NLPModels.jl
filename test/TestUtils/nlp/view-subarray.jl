export view_subarray_nlp

"""
    view_subarray_nlp(nlp)

Check that the API work with views, and that the results is correct.
"""
function view_subarray_nlp(nlp)
  @testset "Test view subarray of NLPs" begin
    n, m = nlp.meta.nvar, nlp.meta.ncon
    N = 2n
    Vidxs = [1:2:N, collect(N:-2:1)]
    Cidxs = if m > 0
      N = 2m
      [1:2:N, collect(N:-2:1)]
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
      #vals1 = hess_coord(nlp, x[I])
      #vals2 = hess_coord(nlp, xv)
      #@test vals1 ≈ vals2

      if m > 0
        for foo in (cons, jac)
          @test foo(nlp, x[I]) ≈ foo(nlp, xv)
        end
        vals1 = jac_coord(nlp, x[I])
        vals2 = jac_coord(nlp, xv)
        @test vals1 ≈ vals2
      end

      for J = Cidxs
        yv = @view y[J]
        @test hess(nlp, x[I], y[J]) ≈ hess(nlp, xv, yv)
        yv = @view y[J]
        #vals1 = hess_coord(nlp, x[I], y[J])
        #vals2 = hess_coord(nlp, xv, yv)
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
          @test hprod(nlp, x[I], y[P], v[J]) ≈ hprod(nlp, xv, yv, vv)
          hprod!(nlp, x[I], y[P], v[J],  hv)
          hprod!(nlp, x[I], y[P], v[J], hvv); @test hv ≈ hv2[K]
          hprod!(nlp,   xv,   yv,   vv, hvv); @test hv ≈ hv2[K]
          hprod!(nlp,   xv,   yv,   vv,  hv); @test hv ≈ hv2[K]
        end
      end
    end
  end
end