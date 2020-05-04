export LLSModel

"""
    nls = LLSModel(A, b; lvar, uvar, C, lcon, ucon)

Creates a Linear Least Squares model ``\\tfrac{1}{2}\\|Ax - b\\|^2`` with optional bounds
`lvar ≦ x ≦ uvar` and optional linear constraints `lcon ≦ Cx ≦ ucon`.
This problem is a nonlinear least-squares problem with residual given by ``F(x) = Ax - b``.
"""
mutable struct LLSModel <: AbstractNLSModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  counters :: NLSCounters

  Arows :: Vector{Int}
  Acols :: Vector{Int}
  Avals :: Vector
  b :: AbstractVector
  Crows :: Vector{Int}
  Ccols :: Vector{Int}
  Cvals :: Vector
end

show_header(io :: IO, nls :: LLSModel) = println(io, "LLSModel - Linear least-squares model")

function LLSModel(A :: AbstractMatrix, b :: AbstractVector;
                  x0 :: AbstractVector = zeros(size(A,2)),
                  lvar :: AbstractVector = fill(-Inf, size(A, 2)),
                  uvar :: AbstractVector = fill(Inf, size(A, 2)),
                  C :: AbstractMatrix  = Matrix{Float64}(undef, 0, 0),
                  lcon :: AbstractVector = Float64[],
                  ucon :: AbstractVector = Float64[],
                  y0 :: AbstractVector = zeros(size(C,1)),
                  name :: String = "generic-LLSModel"
                 )
  nvar = size(A, 2)
  Arows, Acols, Avals = if A isa AbstractSparseMatrix
    findnz(A)
  else
    m, n = size(A)
    I = ((i,j) for i = 1:m, j = 1:n)
    getindex.(I, 1)[:], getindex.(I, 2)[:], A[:]
  end
  Crows, Ccols, Cvals = if C isa AbstractSparseMatrix
    findnz(C)
  else
    m, n = size(C)
    I = ((i,j) for i = 1:m, j = 1:n)
    getindex.(I, 1)[:], getindex.(I, 2)[:], C[:]
  end
  LLSModel(Arows, Acols, Avals, nvar, b, x0=x0, lvar=lvar, uvar=uvar,
           Crows=Crows, Ccols=Ccols, Cvals=Cvals, lcon=lcon, ucon=ucon, y0=y0, name=name)
end

function LLSModel(Arows :: AbstractVector{<: Integer},
                  Acols :: AbstractVector{<: Integer},
                  Avals :: AbstractVector,
                  nvar :: Integer,
                  b :: AbstractVector;
                  x0 :: AbstractVector = zeros(nvar),
                  lvar :: AbstractVector = fill(-Inf, nvar),
                  uvar :: AbstractVector = fill(Inf, nvar),
                  Crows :: AbstractVector{<: Integer} = Int[],
                  Ccols :: AbstractVector{<: Integer} = Int[],
                  Cvals :: AbstractVector = Float64[],
                  lcon :: AbstractVector = Float64[],
                  ucon :: AbstractVector = Float64[],
                  y0 :: AbstractVector = zeros(length(lcon)),
                  name :: String = "generic-LLSModel"
                 )

  nequ = length(b)
  ncon = length(lcon)
  if !(ncon == length(ucon) == length(y0))
    error("The length of lcon, ucon and y0 must be the same")
  end
  nnzjF = length(Avals)
  if !(nnzjF == length(Arows) == length(Acols))
    error("The length of Arows, Acols and Avals must be the same")
  end
  nnzj = length(Cvals)
  if !(nnzj == length(Crows) == length(Ccols))
    error("The length of Crows, Ccols and Cvals must be the same")
  end

  meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, y0=y0, lin=1:ncon,
                      nln=Int[], lcon=lcon, ucon=ucon, nnzj=nnzj, nnzh=0, name=name)

  nls_meta = NLSMeta(nequ, nvar, nnzj=nnzjF, nnzh=0, lin=1:nequ, nln=Int[])

  return LLSModel(meta, nls_meta, NLSCounters(), Arows, Acols, Avals, b, Crows, Ccols, Cvals)
end

function residual!(nls :: LLSModel, x :: AbstractVector, Fx :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ Fx
  increment!(nls, :neval_residual)
  coo_prod!(nls.Arows, nls.Acols, nls.Avals, x, Fx)
  Fx .-= nls.b
  return Fx
end

function jac_structure_residual!(nls :: LLSModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck nls.nls_meta.nnzj rows
  @lencheck nls.nls_meta.nnzj cols
  rows .= nls.Arows
  cols .= nls.Acols
  return rows, cols
end

function jac_coord_residual!(nls :: LLSModel, x :: AbstractVector, vals :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nnzj vals
  increment!(nls, :neval_jac_residual)
  vals .= nls.Avals
  return vals
end

function jprod_residual!(nls :: LLSModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.meta.nvar v
  @lencheck nls.nls_meta.nequ Jv
  increment!(nls, :neval_jprod_residual)
  coo_prod!(nls.Arows, nls.Acols, nls.Avals, v, Jv)
  return Jv
end

function jtprod_residual!(nls :: LLSModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ v
  @lencheck nls.meta.nvar Jtv
  increment!(nls, :neval_jtprod_residual)
  coo_prod!(nls.Acols, nls.Arows, nls.Avals, v, Jtv)
  return Jtv
end

function hess_residual(nls :: LLSModel, x :: AbstractVector, v :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ v
  increment!(nls, :neval_hess_residual)
  n = nls.meta.nvar
  return spzeros(n, n)
end

function hess_structure_residual!(nls :: LLSModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck 0 rows
  @lencheck 0 cols
  return rows, cols
end

function hess_coord_residual!(nls :: LLSModel, x :: AbstractVector, v :: AbstractVector, vals :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ v
  @lencheck 0 vals
  increment!(nls, :neval_hess_residual)
  return vals
end

function jth_hess_residual(nls :: LLSModel, x :: AbstractVector, i :: Int)
  @lencheck nls.meta.nvar x
  increment!(nls, :neval_jhess_residual)
  n = nls.meta.nvar
  return spzeros(n, n)
end

function hprod_residual!(nls :: LLSModel, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  @lencheck nls.meta.nvar x v Hiv
  increment!(nls, :neval_hprod_residual)
  fill!(Hiv, zero(eltype(x)))
  return Hiv
end

function cons!(nls :: LLSModel, x :: AbstractVector, c :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.meta.ncon c
  increment!(nls, :neval_cons)
  coo_prod!(nls.Crows, nls.Ccols, nls.Cvals, x, c)
  return c
end

function jac_structure!(nls :: LLSModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck nls.meta.nnzj rows cols
  rows .= nls.Crows
  cols .= nls.Ccols
  return rows, cols
end

function jac_coord!(nls :: LLSModel, x :: AbstractVector, vals :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.meta.nnzj vals
  increment!(nls, :neval_jac)
  vals .= nls.Cvals
  return vals
end

function jprod!(nls :: LLSModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck nls.meta.nvar x v
  @lencheck nls.meta.ncon Jv
  increment!(nls, :neval_jprod)
  coo_prod!(nls.Crows, nls.Ccols, nls.Cvals, v, Jv)
  return Jv
end

function jtprod!(nls :: LLSModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck nls.meta.nvar x Jtv
  @lencheck nls.meta.ncon v
  increment!(nls, :neval_jtprod)
  coo_prod!(nls.Ccols, nls.Crows, nls.Cvals, v, Jtv)
  return Jtv
end

function hprod!(nls :: LLSModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight = 1.0)
  @lencheck nls.meta.nvar x v Hv
  increment!(nls, :neval_hprod)
  Av = zeros(nls.nls_meta.nequ)
  coo_prod!(nls.Arows, nls.Acols, nls.Avals, v, Av)
  coo_prod!(nls.Acols, nls.Arows, nls.Avals, Av, Hv)
  Hv .*= obj_weight
  return Hv
end

function hprod!(nls :: LLSModel, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight = 1.0)
  @lencheck nls.meta.nvar x v Hv
  @lencheck nls.meta.ncon y
  hprod!(nls, x, v, Hv, obj_weight=obj_weight)
end
