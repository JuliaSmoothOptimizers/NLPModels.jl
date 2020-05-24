using NLPModels: increment!

function nlslc_autodiff()

  A = [1 2; 3 4]
  b = [5; 6]
  B = diagm([3 * i for i = 3:5])
  c = [1; 2; 3]
  C = [0 -2; 4 0]
  d = [1; -1]

  x0 = zeros(15)
  F(x) = [x[i]^2 - i^2 for i=1:15]
  con(x) = [15 * x[15];
            c' * x[10:12];
            d' * x[13:14];
            b' * x[8:9];
            C * x[6:7];
            A * x[1:2];
            B * x[3:5]]

  lcon = [22.0; 1.0; -Inf; -11.0; -d;            -b; -Inf * ones(3)]
  ucon = [22.0; Inf; 16.0;   9.0; -d; Inf * ones(2);              c]

  return ADNLSModel(F, x0, 15, con, lcon, ucon, name="nlslincon_autodiff")
end

mutable struct NLSLC <: AbstractNLSModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  counters :: NLSCounters
end

function NLSLC()
  meta = NLPModelMeta(15, nnzj=17, ncon=11, x0=zeros(15), lcon = [22.0; 1.0; -Inf; -11.0; -1.0; 1.0; -5.0; -6.0; -Inf * ones(3)], ucon=[22.0; Inf; 16.0; 9.0; -1.0; 1.0; Inf * ones(2); 1.0; 2.0; 3.0], name="NLSLINCON")
  nls_meta = NLSMeta(15, 15, nnzj=15, nnzh=15)

  return NLSLC(meta, nls_meta, NLSCounters())
end

function NLPModels.residual!(nls :: NLSLC, x :: AbstractVector, Fx :: AbstractVector)
  @lencheck 15 x Fx
  increment!(nls, :neval_residual)
  Fx .= [x[i]^2 - i^2 for i = 1:nls.nls_meta.nequ]
  return Fx
end

function NLPModels.jac_structure_residual!(nls :: NLSLC, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck 15 rows cols
  for i = 1:nls.nls_meta.nnzj
    rows[i] = i
    cols[i] = i
  end
  return rows, cols
end

function NLPModels.jac_coord_residual!(nls :: NLSLC, x :: AbstractVector, vals :: AbstractVector)
  @lencheck 15 x vals
  increment!(nls, :neval_jac_residual)
  vals .= [2 * x[i] for i = 1:nls.nls_meta.nnzj]
  return vals
end

function NLPModels.jprod_residual!(nls :: NLSLC, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck 15 x v Jv
  increment!(nls, :neval_jprod_residual)
  Jv .= [2 * x[i] * v[i] for i = 1:nls.nls_meta.nnzj]
  return Jv
end

function NLPModels.jtprod_residual!(nls :: NLSLC, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck 15 x v Jtv
  increment!(nls, :neval_jtprod_residual)
  Jtv .= [2 * x[i] * v[i] for i = 1:nls.nls_meta.nnzj]
  return Jtv
end

function NLPModels.hess_structure_residual!(nls :: NLSLC, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck 15 rows cols
  for i = 1:nls.nls_meta.nnzh
    rows[i] = i
    cols[i] = i
  end
  return rows, cols
end

function NLPModels.hess_coord_residual!(nls :: NLSLC, x :: AbstractVector, v :: AbstractVector, vals :: AbstractVector)
  @lencheck 15 x v vals
  increment!(nls, :neval_hess_residual)
  vals .= [2 * v[i] for i = 1:nls.nls_meta.nnzh]
  return vals
end

function NLPModels.hprod_residual!(nls :: NLSLC, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  @lencheck 15 x v Hiv
  increment!(nls, :neval_hprod_residual)
  Hiv .= zero(eltype(x))
  Hiv[i] = 2 * v[i]
  return Hiv
end

function NLPModels.cons!(nls :: NLSLC, x :: AbstractVector, cx :: AbstractVector)
  @lencheck 15 x
  @lencheck 11 cx
  increment!(nls, :neval_cons)
  cx .= [15 * x[15];
        [1; 2; 3]' * x[10:12];
        [1; -1]' * x[13:14];
        [5; 6]' * x[8:9];
        [0 -2; 4 0] * x[6:7];
        [1 2; 3 4] * x[1:2];
        diagm([3 * i for i = 3:5]) * x[3:5]]
  return cx
end

function NLPModels.jac_structure!(nls :: NLSLC, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck 17 rows cols
  rows .= [ 1,  2,  2,  2,  3,  3, 4, 4, 5, 6, 7, 7, 8, 8, 9, 10, 11]
  cols .= [15, 10, 11, 12, 13, 14, 8, 9, 7, 6, 1, 2, 1, 2, 3,  4,  5]
  return rows, cols
end

function NLPModels.jac_coord!(nls :: NLSLC, x :: AbstractVector, vals :: AbstractVector)
  @lencheck 15 x
  @lencheck 17 vals
  increment!(nls, :neval_jac)
  vals .= eltype(x).([15, 1, 2, 3, 1, -1, 5, 6, -2, 4, 1, 2, 3, 4, 9, 12, 15])
  return vals
end

function NLPModels.jprod!(nls :: NLSLC, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck 15 x v
  @lencheck 11 Jv
  increment!(nls, :neval_jprod)
  Jv[1]    = 15 * v[15]
  Jv[2]    = [1; 2; 3]' * v[10:12]
  Jv[3]    = [1; -1]' * v[13:14]
  Jv[4]    = [5; 6]' * v[8:9]
  Jv[5:6]  = [0 -2; 4 0] * v[6:7]
  Jv[7:8]  = [1.0 2; 3 4] * v[1:2]
  Jv[9:11] = diagm([3 * i for i = 3:5]) * v[3:5]
  return Jv
end

function NLPModels.jtprod!(nls :: NLSLC, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck 15 x Jtv
  @lencheck 11 v
  increment!(nls, :neval_jtprod)
  Jtv[1]  = 1 * v[7] + 3 * v[8]
  Jtv[2]  = 2 * v[7] + 4 * v[8]
  Jtv[3]  = 9 * v[9]
  Jtv[4]  = 12 * v[10]
  Jtv[5]  = 15 * v[11]
  Jtv[6]  = 4 * v[6]
  Jtv[7]  = -2 * v[5]
  Jtv[8]  = 5 * v[4]
  Jtv[9]  = 6 * v[4]
  Jtv[10] = 1 * v[2]
  Jtv[11] = 2 * v[2]
  Jtv[12] = 3 * v[2]
  Jtv[13] = 1 * v[3]
  Jtv[14] = -1 * v[3]
  Jtv[15] = 15 * v[1]
  return Jtv
end
