export CDS_scaled_add!
function CDS_scaled_add!{TF<:Real,TI<:Integer}(
                  A           ::Array{TF,2},
                  B           ::Array{TF,2},
                  A_offsets   ::Vector{TI},
                  B_offsets   ::Vector{TI},
                  alpha       ::TF
                  )
# Computes A = A + alpha * B for A and B in the compressed diagonal storage format (CDS/DIA)

for k=1:length(B_offsets)
  A_update_col = findin(A_offsets,B_offsets[k])
  if isempty(A_update_col) == true
    error("attempted to update a diagonal in A in CDS storage that does not excist. A and B need to have the same nonzero diagonals")
  end
  A[:,A_update_col] .= A[:,A_update_col] .+ alpha .* B[:,k];
end



end