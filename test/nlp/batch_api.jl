@testset "Batch API" begin
  # Generate models with varying curvature parameter
  n_models = 5
  models = [SimpleNLPModel() for _ = 1:n_models]
  meta = models[1].meta
  n, m = meta.nvar, meta.ncon
  T = eltype(meta.lcon)
  p_values = [T(2 + i) for i = 1:n_models]
  for i = 1:n_models
    models[i].p = p_values[i]
  end
  xs = [randn(n) for _ = 1:n_models]
  ys = [randn(m) for _ = 1:n_models]
  vs = [randn(n) for _ = 1:n_models]
  ws = [randn(m) for _ = 1:n_models]
  gs = [zeros(n) for _ = 1:n_models]
  cs = [zeros(m) for _ = 1:n_models]
  obj_weights = rand(n_models)
  function make_inplace_batch_model()
    base_model = SimpleNLPModel()
    updates = [
      nlp -> (nlp.p = p) for p in p_values
    ]
    return InplaceBatchNLPModel(base_model, updates)
  end

  @test_throws ErrorException InplaceBatchNLPModel(SimpleNLPModel(), [])

  batch_model_builders = [
    ("ForEachBatchNLPModel", () -> ForEachBatchNLPModel(models)),
    ("InplaceBatchNLPModel", () -> make_inplace_batch_model()),
  ]

  for (batch_model_name, build_batch_model) in batch_model_builders
    @testset "$batch_model_name consistency" begin
      bnlp = build_batch_model()

      # Test batch_obj
      batch_fs = batch_obj(bnlp, xs)
      manual_fs = [obj(models[i], xs[i]) for i = 1:n_models]
      @test batch_fs ≈ manual_fs

      # Test batch_grad
      batch_gs = batch_grad(bnlp, xs)
      manual_gs = [grad(models[i], xs[i]) for i = 1:n_models]
      @test batch_gs ≈ manual_gs

      # Test batch_grad!
      batch_grad!(bnlp, xs, gs)
      manual_gs = [grad!(models[i], xs[i], zeros(n)) for i = 1:n_models]
      @test gs ≈ manual_gs

      # Test batch_objgrad
      batch_fs, batch_gs = batch_objgrad(bnlp, xs)
      manual_fs = [obj(models[i], xs[i]) for i = 1:n_models]
      manual_gs = [grad(models[i], xs[i]) for i = 1:n_models]
      @test batch_fs ≈ manual_fs
      @test batch_gs ≈ manual_gs

      # Test batch_objgrad!
      batch_fs, batch_gs = batch_objgrad!(bnlp, xs, gs)
      manual_fs = [obj(models[i], xs[i]) for i = 1:n_models]
      manual_gs = [grad!(models[i], xs[i], zeros(n)) for i = 1:n_models]
      @test batch_fs ≈ manual_fs
      @test batch_gs ≈ manual_gs

      # Test batch_cons
      batch_cs = batch_cons(bnlp, xs)
      manual_cs = [cons(models[i], xs[i]) for i = 1:n_models]
      @test batch_cs ≈ manual_cs

      # Test batch_cons!
      batch_cons!(bnlp, xs, cs)
      manual_cs = [cons!(models[i], xs[i], zeros(m)) for i = 1:n_models]
      @test cs ≈ manual_cs

      # Test batch_cons_lin
      batch_cs_lin = batch_cons_lin(bnlp, xs)
      manual_cs_lin = [cons_lin(models[i], xs[i]) for i = 1:n_models]
      @test batch_cs_lin ≈ manual_cs_lin

      # Test batch_cons_lin!
      cs_lin = [zeros(meta.nlin) for _ = 1:n_models]
      batch_cons_lin!(bnlp, xs, cs_lin)
      manual_cs_lin = [cons_lin!(models[i], xs[i], zeros(meta.nlin)) for i = 1:n_models]
      @test cs_lin ≈ manual_cs_lin

      # Test batch_cons_nln
      batch_cs_nln = batch_cons_nln(bnlp, xs)
      manual_cs_nln = [cons_nln(models[i], xs[i]) for i = 1:n_models]
      @test batch_cs_nln ≈ manual_cs_nln

      # Test batch_cons_nln!
      cs_nln = [zeros(meta.nnln) for _ = 1:n_models]
      batch_cons_nln!(bnlp, xs, cs_nln)
      manual_cs_nln = [cons_nln!(models[i], xs[i], zeros(meta.nnln)) for i = 1:n_models]
      @test cs_nln ≈ manual_cs_nln

      # Test batch_objcons
      batch_fs, batch_cs = batch_objcons(bnlp, xs)
      manual_fs = [obj(models[i], xs[i]) for i = 1:n_models]
      manual_cs = [cons(models[i], xs[i]) for i = 1:n_models]
      @test batch_fs ≈ manual_fs
      @test batch_cs ≈ manual_cs

      # Test batch_objcons!
      batch_fs, batch_cs = batch_objcons!(bnlp, xs, cs)
      manual_fs = [obj(models[i], xs[i]) for i = 1:n_models]
      manual_cs = [cons!(models[i], xs[i], zeros(m)) for i = 1:n_models]
      @test batch_fs ≈ manual_fs
      @test batch_cs ≈ manual_cs

      # Test batch_jac
      batch_jacs = batch_jac(bnlp, xs)
      manual_jacs = [jac(models[i], xs[i]) for i = 1:n_models]
      @test batch_jacs ≈ manual_jacs

      # Test batch_jac_coord
      batch_jac_coords = batch_jac_coord(bnlp, xs)
      manual_jac_coords = [jac_coord(models[i], xs[i]) for i = 1:n_models]
      @test batch_jac_coords ≈ manual_jac_coords

      # Test batch_jac_coord!
      jac_coords = [zeros(meta.nnzj) for _ = 1:n_models]
      batch_jac_coord!(bnlp, xs, jac_coords)
      manual_jac_coords = [jac_coord!(models[i], xs[i], zeros(meta.nnzj)) for i = 1:n_models]
      @test jac_coords ≈ manual_jac_coords

      # Test batch_jac_lin
      batch_jac_lins = batch_jac_lin(bnlp)
      manual_jac_lins = [jac_lin(models[i]) for i = 1:n_models]
      @test batch_jac_lins ≈ manual_jac_lins

      # Test batch_jac_lin_coord
      batch_jac_lin_coords = batch_jac_lin_coord(bnlp)
      manual_jac_lin_coords = [jac_lin_coord(models[i]) for i = 1:n_models]
      @test batch_jac_lin_coords ≈ manual_jac_lin_coords

      # Test batch_jac_lin_coord!
      jac_lin_coords = [zeros(meta.lin_nnzj) for _ = 1:n_models]
      batch_jac_lin_coord!(bnlp, jac_lin_coords)
      manual_jac_lin_coords =
        [jac_lin_coord!(models[i], zeros(meta.lin_nnzj)) for i = 1:n_models]
      @test jac_lin_coords ≈ manual_jac_lin_coords

      # Test batch_jac_nln
      batch_jac_nlns = batch_jac_nln(bnlp, xs)
      manual_jac_nlns = [jac_nln(models[i], xs[i]) for i = 1:n_models]
      @test batch_jac_nlns ≈ manual_jac_nlns

      # Test batch_jac_nln_coord
      batch_jac_nln_coords = batch_jac_nln_coord(bnlp, xs)
      manual_jac_nln_coords = [jac_nln_coord(models[i], xs[i]) for i = 1:n_models]
      @test batch_jac_nln_coords ≈ manual_jac_nln_coords

      # Test batch_jac_nln_coord!
      jac_nln_coords = [zeros(meta.nln_nnzj) for _ = 1:n_models]
      batch_jac_nln_coord!(bnlp, xs, jac_nln_coords)
      manual_jac_nln_coords =
        [jac_nln_coord!(models[i], xs[i], zeros(meta.nln_nnzj)) for i = 1:n_models]
      @test jac_nln_coords ≈ manual_jac_nln_coords

      # Test batch_jprod
      batch_jprods = batch_jprod(bnlp, xs, vs)
      manual_jprods = [jprod(models[i], xs[i], vs[i]) for i = 1:n_models]
      @test batch_jprods ≈ manual_jprods

      # Test batch_jprod!
      jprods = [zeros(m) for _ = 1:n_models]
      batch_jprod!(bnlp, xs, vs, jprods)
      manual_jprods = [jprod!(models[i], xs[i], vs[i], zeros(m)) for i = 1:n_models]
      @test jprods ≈ manual_jprods

      # Test batch_jtprod
      batch_jtprods = batch_jtprod(bnlp, xs, ws)
      manual_jtprods = [jtprod(models[i], xs[i], ws[i]) for i = 1:n_models]
      @test batch_jtprods ≈ manual_jtprods

      # Test batch_jtprod!
      jtprods = [zeros(n) for _ = 1:n_models]
      batch_jtprod!(bnlp, xs, ws, jtprods)
      manual_jtprods = [jtprod!(models[i], xs[i], ws[i], zeros(n)) for i = 1:n_models]
      @test jtprods ≈ manual_jtprods

      # Test batch_jprod_lin
      batch_jprod_lins = batch_jprod_lin(bnlp, vs)
      manual_jprod_lins = [jprod_lin(models[i], vs[i]) for i = 1:n_models]
      @test batch_jprod_lins ≈ manual_jprod_lins

      # Test batch_jprod_lin!
      jprod_lins = [zeros(meta.nlin) for _ = 1:n_models]
      batch_jprod_lin!(bnlp, vs, jprod_lins)
      manual_jprod_lins = [jprod_lin!(models[i], vs[i], zeros(meta.nlin)) for i = 1:n_models]
      @test jprod_lins ≈ manual_jprod_lins

      # Test batch_jtprod_lin
      ws_lin = [ws[i][1:(meta.nlin)] for i = 1:n_models]
      batch_jtprod_lins = batch_jtprod_lin(bnlp, ws_lin)
      manual_jtprod_lins = [jtprod_lin(models[i], ws_lin[i]) for i = 1:n_models]
      @test batch_jtprod_lins ≈ manual_jtprod_lins

      # Test batch_jtprod_lin!
      jtprod_lins = [zeros(n) for _ = 1:n_models]
      batch_jtprod_lin!(bnlp, ws_lin, jtprod_lins)
      manual_jtprod_lins = [jtprod_lin!(models[i], ws_lin[i], zeros(n)) for i = 1:n_models]
      @test jtprod_lins ≈ manual_jtprod_lins

      # Test batch_jprod_nln
      batch_jprod_nlns = batch_jprod_nln(bnlp, xs, vs)
      manual_jprod_nlns = [jprod_nln(models[i], xs[i], vs[i]) for i = 1:n_models]
      @test batch_jprod_nlns ≈ manual_jprod_nlns

      # Test batch_jprod_nln!
      jprod_nlns = [zeros(meta.nnln) for _ = 1:n_models]
      batch_jprod_nln!(bnlp, xs, vs, jprod_nlns)
      manual_jprod_nlns =
        [jprod_nln!(models[i], xs[i], vs[i], zeros(meta.nnln)) for i = 1:n_models]
      @test jprod_nlns ≈ manual_jprod_nlns

      # Test batch_jtprod_nln
      ws_nln = [ws[i][(meta.nlin + 1):end] for i = 1:n_models]
      batch_jtprod_nlns = batch_jtprod_nln(bnlp, xs, ws_nln)
      manual_jtprod_nlns = [jtprod_nln(models[i], xs[i], ws_nln[i]) for i = 1:n_models]
      @test batch_jtprod_nlns ≈ manual_jtprod_nlns

      # Test batch_jtprod_nln!
      jtprod_nlns = [zeros(n) for _ = 1:n_models]
      batch_jtprod_nln!(bnlp, xs, ws_nln, jtprod_nlns)
      manual_jtprod_nlns = [jtprod_nln!(models[i], xs[i], ws_nln[i], zeros(n)) for i = 1:n_models]
      @test jtprod_nlns ≈ manual_jtprod_nlns

      # Test batch_hess with obj_weights (without y)
      batch_hesses = batch_hess(bnlp, xs; obj_weights = obj_weights)
      manual_hesses = [hess(models[i], xs[i]; obj_weight = obj_weights[i]) for i = 1:n_models]
      @test batch_hesses ≈ manual_hesses

      # Test batch_hess with obj_weights (with y)
      batch_hesses = batch_hess(bnlp, xs, ys; obj_weights = obj_weights)
      manual_hesses =
        [hess(models[i], xs[i], ys[i]; obj_weight = obj_weights[i]) for i = 1:n_models]
      @test batch_hesses ≈ manual_hesses

      # Test batch_hess_coord with obj_weights (without y)
      batch_hess_coords = batch_hess_coord(bnlp, xs; obj_weights = obj_weights)
      manual_hess_coords =
        [hess_coord(models[i], xs[i]; obj_weight = obj_weights[i]) for i = 1:n_models]
      @test batch_hess_coords ≈ manual_hess_coords

      # Test batch_hess_coord with obj_weights (with y)
      batch_hess_coords = batch_hess_coord(bnlp, xs, ys; obj_weights = obj_weights)
      manual_hess_coords =
        [hess_coord(models[i], xs[i], ys[i]; obj_weight = obj_weights[i]) for i = 1:n_models]
      @test batch_hess_coords ≈ manual_hess_coords

      # Test batch_hess_coord! with obj_weights (without y)
      hess_coords = [zeros(meta.nnzh) for _ = 1:n_models]
      batch_hess_coord!(bnlp, xs, hess_coords; obj_weights = obj_weights)
      manual_hess_coords = [
        hess_coord!(models[i], xs[i], zeros(meta.nnzh); obj_weight = obj_weights[i]) for
        i = 1:n_models
      ]
      @test hess_coords ≈ manual_hess_coords

      # Test batch_hess_coord! with obj_weights (with y)
      hess_coords = [zeros(meta.nnzh) for _ = 1:n_models]
      batch_hess_coord!(bnlp, xs, ys, hess_coords; obj_weights = obj_weights)
      manual_hess_coords = [
        hess_coord!(models[i], xs[i], ys[i], zeros(meta.nnzh); obj_weight = obj_weights[i])
        for i = 1:n_models
      ]
      @test hess_coords ≈ manual_hess_coords

      # Test batch_hprod with obj_weights (without y)
      batch_hprods = batch_hprod(bnlp, xs, vs; obj_weights = obj_weights)
      manual_hprods =
        [hprod(models[i], xs[i], vs[i]; obj_weight = obj_weights[i]) for i = 1:n_models]
      @test batch_hprods ≈ manual_hprods

      # Test batch_hprod with obj_weights (with y)
      batch_hprods = batch_hprod(bnlp, xs, ys, vs; obj_weights = obj_weights)
      manual_hprods =
        [hprod(models[i], xs[i], ys[i], vs[i]; obj_weight = obj_weights[i]) for i = 1:n_models]
      @test batch_hprods ≈ manual_hprods

      # Test batch_hprod! with obj_weights (without y)
      hprods = [zeros(n) for _ = 1:n_models]
      batch_hprod!(bnlp, xs, vs, hprods; obj_weights = obj_weights)
      manual_hprods =
        [hprod!(models[i], xs[i], vs[i], zeros(n); obj_weight = obj_weights[i]) for i = 1:n_models]
      @test hprods ≈ manual_hprods

      # Test batch_hprod! with obj_weights (with y)
      hprods = [zeros(n) for _ = 1:n_models]
      batch_hprod!(bnlp, xs, ys, vs, hprods; obj_weights = obj_weights)
      manual_hprods = [
        hprod!(models[i], xs[i], ys[i], vs[i], zeros(n); obj_weight = obj_weights[i]) for
        i = 1:n_models
      ]
      @test hprods ≈ manual_hprods

      if isa(bnlp, ForEachBatchNLPModel) # NOTE: excluding InplaceBatchNLPModel
        # Test batch_hess_op with obj_weights (without y)
        batch_hess_ops = batch_hess_op(bnlp, xs; obj_weights = obj_weights)
        manual_hess_ops = [
          hess_op(models[i], xs[i]; obj_weight = obj_weights[i]) for i = 1:n_models
        ]
        for i = 1:n_models
          @test batch_hess_ops[i] * vs[i] ≈ manual_hess_ops[i] * vs[i]
        end

        # Test batch_hess_op with obj_weights (with y)
        batch_hess_ops = batch_hess_op(bnlp, xs, ys; obj_weights = obj_weights)
        manual_hess_ops = [
          hess_op(models[i], xs[i], ys[i]; obj_weight = obj_weights[i]) for i = 1:n_models
        ]
        for i = 1:n_models
          @test batch_hess_ops[i] * vs[i] ≈ manual_hess_ops[i] * vs[i]
        end

        # Test batch_hess_op! with obj_weights (without y)
        hvs = [zeros(n) for _ = 1:n_models]
        batch_hess_ops = batch_hess_op!(bnlp, xs, hvs; obj_weights = obj_weights)
        manual_hess_ops = [
          hess_op!(models[i], xs[i], zeros(n); obj_weight = obj_weights[i]) for i = 1:n_models
        ]
        for i = 1:n_models
          @test batch_hess_ops[i] * vs[i] ≈ manual_hess_ops[i] * vs[i]
        end

        # Test batch_hess_op! with obj_weights (with y)
        hvs = [zeros(n) for _ = 1:n_models]
        batch_hess_ops = batch_hess_op!(bnlp, xs, ys, hvs; obj_weights = obj_weights)
        manual_hess_ops = [
          hess_op!(models[i], xs[i], ys[i], zeros(n); obj_weight = obj_weights[i]) for
          i = 1:n_models
        ]
        for i = 1:n_models
          @test batch_hess_ops[i] * vs[i] ≈ manual_hess_ops[i] * vs[i]
        end
      else
        @test_throws ErrorException batch_hess_op(bnlp, xs; obj_weights = obj_weights)
        @test_throws ErrorException batch_hess_op(bnlp, xs, ys; obj_weights = obj_weights)
        @test_throws ErrorException batch_hess_op!(bnlp, xs, [zeros(n) for _ = 1:n_models];
                                                   obj_weights = obj_weights)
        @test_throws ErrorException batch_hess_op!(
          bnlp,
          xs,
          ys,
          [zeros(n) for _ = 1:n_models];
          obj_weights = obj_weights,
        )
      end

      # Test batch_jth_con
      j = 1
      batch_jth_cons = batch_jth_con(bnlp, xs, j)
      manual_jth_cons = [jth_con(models[i], xs[i], j) for i = 1:n_models]
      @test batch_jth_cons ≈ manual_jth_cons

      # Test batch_jth_congrad
      batch_jth_congrads = batch_jth_congrad(bnlp, xs, j)
      manual_jth_congrads = [jth_congrad(models[i], xs[i], j) for i = 1:n_models]
      @test batch_jth_congrads ≈ manual_jth_congrads

      # Test batch_jth_congrad!
      jth_congrads = [zeros(n) for _ = 1:n_models]
      batch_jth_congrad!(bnlp, xs, j, jth_congrads)
      manual_jth_congrads = [jth_congrad!(models[i], xs[i], j, zeros(n)) for i = 1:n_models]
      @test jth_congrads ≈ manual_jth_congrads

      # Test batch_jth_sparse_congrad
      batch_jth_sparse_congrads = batch_jth_sparse_congrad(bnlp, xs, j)
      manual_jth_sparse_congrads = [jth_sparse_congrad(models[i], xs[i], j) for i = 1:n_models]
      @test batch_jth_sparse_congrads ≈ manual_jth_sparse_congrads

      # Test batch_jth_hess_coord
      batch_jth_hess_coords = batch_jth_hess_coord(bnlp, xs, j)
      manual_jth_hess_coords = [jth_hess_coord(models[i], xs[i], j) for i = 1:n_models]
      @test batch_jth_hess_coords ≈ manual_jth_hess_coords

      # Test batch_jth_hess_coord!
      jth_hess_coords = [zeros(meta.nnzh) for _ = 1:n_models]
      batch_jth_hess_coord!(bnlp, xs, j, jth_hess_coords)
      manual_jth_hess_coords =
        [jth_hess_coord!(models[i], xs[i], j, zeros(meta.nnzh)) for i = 1:n_models]
      @test jth_hess_coords ≈ manual_jth_hess_coords

      # Test batch_jth_hess
      batch_jth_hesses = batch_jth_hess(bnlp, xs, j)
      manual_jth_hesses = [jth_hess(models[i], xs[i], j) for i = 1:n_models]
      @test batch_jth_hesses ≈ manual_jth_hesses

      # Test batch_jth_hprod
      batch_jth_hprods = batch_jth_hprod(bnlp, xs, vs, j)
      manual_jth_hprods = [jth_hprod(models[i], xs[i], vs[i], j) for i = 1:n_models]
      @test batch_jth_hprods ≈ manual_jth_hprods

      # Test batch_jth_hprod!
      jth_hprods = [zeros(n) for _ = 1:n_models]
      batch_jth_hprod!(bnlp, xs, vs, j, jth_hprods)
      manual_jth_hprods = [jth_hprod!(models[i], xs[i], vs[i], j, zeros(n)) for i = 1:n_models]
      @test jth_hprods ≈ manual_jth_hprods

      # Test batch_ghjvprod
      batch_ghjvprods = batch_ghjvprod(bnlp, xs, gs, vs)
      manual_ghjvprods = [ghjvprod(models[i], xs[i], gs[i], vs[i]) for i = 1:n_models]
      @test batch_ghjvprods ≈ manual_ghjvprods

      # Test batch_ghjvprod!
      ghjvprods = [zeros(m) for _ = 1:n_models]
      batch_ghjvprod!(bnlp, xs, gs, vs, ghjvprods)
      manual_ghjvprods = [ghjvprod!(models[i], xs[i], gs[i], vs[i], zeros(m)) for i = 1:n_models]
      @test ghjvprods ≈ manual_ghjvprods

      if isa(bnlp, ForEachBatchNLPModel)
        # Test batch_jac_op
        batch_jac_ops = batch_jac_op(bnlp, xs)
        manual_jac_ops = [jac_op(models[i], xs[i]) for i = 1:n_models]
        for i = 1:n_models
          @test batch_jac_ops[i] * vs[i] ≈ manual_jac_ops[i] * vs[i]
          @test batch_jac_ops[i]' * ws[i] ≈ manual_jac_ops[i]' * ws[i]
        end

        # Test batch_jac_op!
        jvs = [zeros(m) for _ = 1:n_models]
        jtvs = [zeros(n) for _ = 1:n_models]
        batch_jac_ops = batch_jac_op!(bnlp, xs, jvs, jtvs)
        manual_jac_ops = [jac_op!(models[i], xs[i], zeros(m), zeros(n)) for i = 1:n_models]
        for i = 1:n_models
          @test batch_jac_ops[i] * vs[i] ≈ manual_jac_ops[i] * vs[i]
          @test batch_jac_ops[i]' * ws[i] ≈ manual_jac_ops[i]' * ws[i]
        end

        # Test batch_jac_lin_op
        batch_jac_lin_ops = batch_jac_lin_op(bnlp)
        manual_jac_lin_ops = [jac_lin_op(models[i]) for i = 1:n_models]
        ws_lin_vec = ws[1][1:(meta.nlin)]
        for i = 1:n_models
          @test batch_jac_lin_ops[i] * vs[i] ≈ manual_jac_lin_ops[i] * vs[i]
          @test batch_jac_lin_ops[i]' * ws_lin_vec ≈ manual_jac_lin_ops[i]' * ws_lin_vec
        end

        # Test batch_jac_lin_op!
        jvs_lin = [zeros(meta.nlin) for _ = 1:n_models]
        jtvs_lin = [zeros(n) for _ = 1:n_models]
        batch_jac_lin_ops = batch_jac_lin_op!(bnlp, jvs_lin, jtvs_lin)
        manual_jac_lin_ops =
          [jac_lin_op!(models[i], zeros(meta.nlin), zeros(n)) for i = 1:n_models]
        for i = 1:n_models
          @test batch_jac_lin_ops[i] * vs[i] ≈ manual_jac_lin_ops[i] * vs[i]
          @test batch_jac_lin_ops[i]' * ws_lin_vec ≈ manual_jac_lin_ops[i]' * ws_lin_vec
        end

        # Test batch_jac_nln_op
        batch_jac_nln_ops = batch_jac_nln_op(bnlp, xs)
        manual_jac_nln_ops = [jac_nln_op(models[i], xs[i]) for i = 1:n_models]
        ws_nln_vec = ws[1][(meta.nlin + 1):end]
        for i = 1:n_models
          @test batch_jac_nln_ops[i] * vs[i] ≈ manual_jac_nln_ops[i] * vs[i]
          @test batch_jac_nln_ops[i]' * ws_nln_vec ≈ manual_jac_nln_ops[i]' * ws_nln_vec
        end

        # Test batch_jac_nln_op!
        jvs_nln = [zeros(meta.nnln) for _ = 1:n_models]
        jtvs_nln = [zeros(n) for _ = 1:n_models]
        batch_jac_nln_ops = batch_jac_nln_op!(bnlp, xs, jvs_nln, jtvs_nln)
        manual_jac_nln_ops =
          [jac_nln_op!(models[i], xs[i], zeros(meta.nnln), zeros(n)) for i = 1:n_models]
        for i = 1:n_models
          @test batch_jac_nln_ops[i] * vs[i] ≈ manual_jac_nln_ops[i] * vs[i]
          @test batch_jac_nln_ops[i]' * ws_nln_vec ≈ manual_jac_nln_ops[i]' * ws_nln_vec
        end
      else
        @test_throws ErrorException batch_jac_op(bnlp, xs)
        @test_throws ErrorException batch_jac_op!(bnlp, xs, [zeros(m) for _ = 1:n_models],
                                                  [zeros(n) for _ = 1:n_models])
        @test_throws ErrorException batch_jac_lin_op(bnlp)
        @test_throws ErrorException batch_jac_lin_op!(bnlp,
                                                      [zeros(meta.nlin) for _ = 1:n_models],
                                                      [zeros(n) for _ = 1:n_models])
        @test_throws ErrorException batch_jac_nln_op(bnlp, xs)
        @test_throws ErrorException batch_jac_nln_op!(bnlp,
                                                      xs,
                                                      [zeros(meta.nnln) for _ = 1:n_models],
                                                      [zeros(n) for _ = 1:n_models])
      end

      # Test batch_varscale, batch_lagscale, batch_conscale
      batch_varscales = batch_varscale(bnlp)
      manual_varscales = [varscale(models[i]) for i = 1:n_models]
      @test batch_varscales ≈ manual_varscales

      batch_lagscales = batch_lagscale(bnlp)
      manual_lagscales = [lagscale(models[i]) for i = 1:n_models]
      @test batch_lagscales ≈ manual_lagscales

      batch_conscales = batch_conscale(bnlp)
      manual_conscales = [conscale(models[i]) for i = 1:n_models]
      @test batch_conscales ≈ manual_conscales

      # Test structure functions
      jac_structures = batch_jac_structure(bnlp)
      manual_jac_structures = [jac_structure(models[i]) for i = 1:n_models]
      @test jac_structures == manual_jac_structures

      jac_lin_structures = batch_jac_lin_structure(bnlp)
      manual_jac_lin_structures = [jac_lin_structure(models[i]) for i = 1:n_models]
      @test jac_lin_structures == manual_jac_lin_structures

      jac_nln_structures = batch_jac_nln_structure(bnlp)
      manual_jac_nln_structures = [jac_nln_structure(models[i]) for i = 1:n_models]
      @test jac_nln_structures == manual_jac_nln_structures

      hess_structures = batch_hess_structure(bnlp)
      manual_hess_structures = [hess_structure(models[i]) for i = 1:n_models]
      @test hess_structures == manual_hess_structures

      rowss = [copy(manual_jac_structures[i][1]) for i = 1:n_models]
      colss = [copy(manual_jac_structures[i][2]) for i = 1:n_models]
      foreach(r -> fill!(r, 0), rowss)
      foreach(c -> fill!(c, 0), colss)
      batch_jac_structure!(bnlp, rowss, colss)
      for i = 1:n_models
        @test rowss[i] == manual_jac_structures[i][1]
        @test colss[i] == manual_jac_structures[i][2]
      end

      rowss = [copy(manual_jac_lin_structures[i][1]) for i = 1:n_models]
      colss = [copy(manual_jac_lin_structures[i][2]) for i = 1:n_models]
      foreach(r -> fill!(r, 0), rowss)
      foreach(c -> fill!(c, 0), colss)
      batch_jac_lin_structure!(bnlp, rowss, colss)
      for i = 1:n_models
        @test rowss[i] == manual_jac_lin_structures[i][1]
        @test colss[i] == manual_jac_lin_structures[i][2]
      end

      rowss = [copy(manual_jac_nln_structures[i][1]) for i = 1:n_models]
      colss = [copy(manual_jac_nln_structures[i][2]) for i = 1:n_models]
      foreach(r -> fill!(r, 0), rowss)
      foreach(c -> fill!(c, 0), colss)
      batch_jac_nln_structure!(bnlp, rowss, colss)
      for i = 1:n_models
        @test rowss[i] == manual_jac_nln_structures[i][1]
        @test colss[i] == manual_jac_nln_structures[i][2]
      end

      rowss = [copy(manual_hess_structures[i][1]) for i = 1:n_models]
      colss = [copy(manual_hess_structures[i][2]) for i = 1:n_models]
      foreach(r -> fill!(r, 0), rowss)
      foreach(c -> fill!(c, 0), colss)
      batch_hess_structure!(bnlp, rowss, colss)
      for i = 1:n_models
        @test rowss[i] == manual_hess_structures[i][1]
        @test colss[i] == manual_hess_structures[i][2]
      end
    end
  end
end
