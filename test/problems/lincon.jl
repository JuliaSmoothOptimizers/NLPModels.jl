function lincon_autodiff()

  A = [1 2; 3 4]
  b = [5; 6]
  B = diagm([3 * i for i = 3:5])
  c = [1; 2; 3]
  C = [0 -2; 4 0]
  d = [1; -1]

  x0 = zeros(15)
  f(x) = sum(i + x[i]^4 for i = 1:15)
  con(x) = [15 * x[15];
            c' * x[10:12];
            d' * x[13:14];
            b' * x[8:9];
            C * x[6:7];
            A * x[1:2];
            B * x[3:5]]

  lcon = [22.0; 1.0; -Inf; -11.0; -d;            -b; -Inf * ones(3)]
  ucon = [22.0; Inf; 16.0;   9.0; -d; Inf * ones(2);              c]

  return ADNLPModel(f, x0, con, lcon, ucon, name="lincon_autodiff")
end

mutable struct LINCON <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
end

function LINCON()
  meta = NLPModelMeta(15, nnzh=15, nnzj=17, ncon=11, x0=zeros(15), lcon = [22.0; 1.0; -Inf; -11.0; -1.0; 1.0; -5.0; -6.0; -Inf * ones(3)], ucon=[22.0; Inf; 16.0; 9.0; -1.0; 1.0; Inf * ones(2); 1.0; 2.0; 3.0], name="LINCON_manual")

  return LINCON(meta, Counters())
end

function NLPModels.obj(nlp :: LINCON, x :: AbstractVector)
  @lencheck 15 x
  increment!(nlp, :neval_obj)
  return sum(i + x[i]^4 for i = 1:nlp.meta.nvar)
end

function NLPModels.grad!(nlp :: LINCON, x :: AbstractVector, gx :: AbstractVector)
  @lencheck 15 x gx
  increment!(nlp, :neval_grad)
  gx .= [4 * x[i]^3 for i =1:nlp.meta.nvar]
  return gx
end

function NLPModels.hess_structure!(nlp :: LINCON, rows :: AbstractVector{Int}, cols :: AbstractVector{Int})
  @lencheck 15 rows cols
  for i = 1:nlp.meta.nnzh
    rows[i] = i
    cols[i] = i
  end
  return rows, cols
end

function NLPModels.hess_coord!(nlp :: LINCON, x :: AbstractVector{T}, vals :: AbstractVector{T}; obj_weight=one(T)) where T
  @lencheck 15 x vals
  increment!(nlp, :neval_hess)
  for i = 1:nlp.meta.nnzh
    vals[i] = 12 * obj_weight * x[i]^2
  end
  return vals
end

function NLPModels.hess_coord!(nlp :: LINCON, x :: AbstractVector{T}, y :: AbstractVector{T}, vals :: AbstractVector{T}; obj_weight=one(T)) where T
  @lencheck 15 x vals
  @lencheck 11 y
  hess_coord!(nlp, x, vals, obj_weight=obj_weight)
end

function NLPModels.hprod!(nlp :: LINCON, x :: AbstractVector{T}, y :: AbstractVector{T}, v :: AbstractVector{T}, Hv :: AbstractVector{T}; obj_weight=one(T)) where T
  @lencheck 15 x v Hv
  @lencheck 11 y
  increment!(nlp, :neval_hprod)
  for i=1:nlp.meta.nvar
    Hv[i] = 12 * obj_weight * x[i]^2 * v[i]
  end
  return Hv
end

function NLPModels.cons!(nlp :: LINCON, x :: AbstractVector, cx :: AbstractVector)
  @lencheck 15 x
  @lencheck 11 cx
  increment!(nlp, :neval_cons)
  cx .= [15 * x[15];
        [1; 2; 3]' * x[10:12];
        [1; -1]' * x[13:14];
        [5; 6]' * x[8:9];
        [0 -2; 4 0] * x[6:7];
        [1 2; 3 4] * x[1:2];
        diagm([3 * i for i = 3:5]) * x[3:5]]
  return cx
end

function NLPModels.jac_structure!(nlp :: LINCON, rows :: AbstractVector{Int}, cols :: AbstractVector{Int})
  @lencheck 17 rows cols
  rows .= [ 1,  2,  2,  2,  3,  3, 4, 4, 5, 6, 7, 7, 8, 8, 9, 10, 11]
  cols .= [15, 10, 11, 12, 13, 14, 8, 9, 7, 6, 1, 2, 1, 2, 3,  4,  5]
  return rows, cols
end

function NLPModels.jac_coord!(nlp :: LINCON, x :: AbstractVector, vals :: AbstractVector)
  @lencheck 15 x
  @lencheck 17 vals
  increment!(nlp, :neval_jac)
  vals .= eltype(x).([15, 1, 2, 3, 1, -1, 5, 6, -2, 4, 1, 2, 3, 4, 9, 12, 15])
  return vals
end

function NLPModels.jprod!(nlp :: LINCON, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck 15 x v
  @lencheck 11 Jv
  increment!(nlp, :neval_jprod)
  Jv[1]    = 15 * v[15]
  Jv[2]    = [1; 2; 3]' * v[10:12]
  Jv[3]    = [1; -1]' * v[13:14]
  Jv[4]    = [5; 6]' * v[8:9]
  Jv[5:6]  = [0 -2; 4 0] * v[6:7]
  Jv[7:8]  = [1.0 2; 3 4] * v[1:2]
  Jv[9:11] = diagm([3 * i for i = 3:5]) * v[3:5]
  return Jv
end

function NLPModels.jtprod!(nlp :: LINCON, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck 15 x Jtv
  @lencheck 11 v
  increment!(nlp, :neval_jtprod)
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
