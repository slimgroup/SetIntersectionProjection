export Q_update!

"""
update the Q matrix in SparseMatrixCSC or CDS format. Q=sum_{i=1}^p AtA[i]rho[i]
if any of the rho[i] change
"""
function Q_update!(
                  Q            ::Union{Array{TF,2},SparseMatrixCSC{TF,TI}},
                  AtA          ::Union{Vector{Array{TF, 2}}, Vector{SparseMatrixCSC{TF, TI}}, Vector{joAbstractLinearOperator{TF, TF}}, Vector{Union{Array{TF, 2}, SparseMatrixCSC{TF, TI}, joAbstractLinearOperator{TF, TF}}}},
                  set_Prop,
                  rho          ::Vector{TF},
                  ind_updated  ::Vector{TI},
                  log_PARSDMM,
                  i            ::Integer,
                  Q_offsets=[] ::Vector{TI}
                  ) where {TF<:Real,TI<:Integer}

  if isempty(ind_updated) == false
    if typeof(Q)==SparseMatrixCSC{TF,TI}
      for ii=1:length(ind_updated)
          Q = Q + ( AtA[ind_updated[ii]] )*( rho[ind_updated[ii]]-log_PARSDMM.rho[i,ind_updated[ii]] );
      end
    else
      for ii=1:length(ind_updated)
        CDS_scaled_add!(Q,AtA[ind_updated[ii]],Q_offsets,set_Prop.AtA_offsets[ind_updated[ii]],rho[ind_updated[ii]] .- log_PARSDMM.rho[i,ind_updated[ii]] )
      end
    end
  end

return Q
end
