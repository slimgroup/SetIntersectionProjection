export CDS_MVp
function CDS_MVp{TF<:Real,TI<:Integer}(
                        N::Integer,
                        ndiags::Integer,
                        R::Array{TF,2},
                        offset::Vector{TI},
                        x::Vector{TF},
                        y::Vector{TF})
#R is a tall matrix N by ndiagonals, corresponding to a square matrix A
  for i = 1 : ndiags
      d = offset[i]
      r0 = max(1, 1-d)
      r1 = min(N, N-d)
      c0 = max(1, 1+d)
       for r = r0 : r1
         c = r - r0 + c0 #original
       @inbounds y[r] = y[r] + R[r,i] * x[c]#original
      end
  end
  return y
end