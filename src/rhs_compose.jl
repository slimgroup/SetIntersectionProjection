export rhs_compose

"""
form the right-hand-side for the linear system in PARSDMM for the x-minimization
"""
function rhs_compose(
                              rhs         ::Vector{TF},
                              l           ::Union{ Vector{Vector{TF}},DistributedArrays.DArray{Array{TF,1},1,Array{Array{TF,1},1}} },
                              y           ::Union{ Vector{Vector{TF}},DistributedArrays.DArray{Array{TF,1},1,Array{Array{TF,1},1}} },
                              rho         ::Union{ Vector{TF}, DistributedArrays.DArray{TF,1,Array{TF,1}} },
                              TD_OP       ::Union{Vector{Union{SparseMatrixCSC{TF,TI},JOLI.joAbstractLinearOperator{TF,TF}}},DistributedArrays.DArray{Union{JOLI.joAbstractLinearOperator{TF,TF}, SparseMatrixCSC{TF,TI}},1,Array{Union{JOLI.joAbstractLinearOperator{TF,TF}, SparseMatrixCSC{TF,TI}},1}} },
                              p           ::Integer,
                              Blas_active ::Bool,
                              parallel    ::Bool
                              ) where {TF<:Real,TI<:Integer}

if parallel==true
  rhs = @distributed (+) for ii = 1:p
  TD_OP[ii]'*(rho[ii] .* y[ii] .+ l[ii])
  end
else #serial
  fill!(rhs,TF(0));
  if Blas_active #serial with explicit BLAS calls
    for ii=1:p
      BLAS.axpy!(TF(1.0), TD_OP[ii]'*(rho[ii] .* y[ii] .+ l[ii]), rhs)
    end
  else
    for ii=1:p #serial with loop fusion only
      rhs .= rhs .+ TD_OP[ii]'*(rho[ii] .* y[ii].+l[ii])
    end
  end
end #end creating new rhs

return rhs
end #function
