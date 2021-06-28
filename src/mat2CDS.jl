export mat2CDS

"""
Convert sparse and square matrix A to compressed diagonal format (CDS)
Currently not multithreaded
"""
function mat2CDS(A::SparseMatrixCSC{TF,TI}) where {TF<:Real,TI<:Integer}

  # Find all nonzero diagonals
  (i,j,dummy) = findnz(A) #[i,j] = find(A);
  d           = sort(j-i)
  #julia 0.6: offset      = d[find(diff([-Inf;d]))] #matlab:d = d(find(diff([-inf; d])));
  offset      = d[findall(!iszero, diff([-Inf;d]))]
  offset      = convert(Array{TI,1},offset)

  # initialize
  (m,n) = size(A)
  p     = length(offset)
  R     = zeros(TF,m,p)

  #construct tall dense matrix with diagonals of A
  for i=1:length(offset)
    dA = diag(A,offset[i])
    if offset[i]>=0
      R[1:length(dA),i]=dA;
    elseif offset[i]<0
      R[end-length(dA)+1:end,i]=dA;
    end
  end

  return R, offset
end
