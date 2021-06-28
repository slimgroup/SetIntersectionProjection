export get_discrete_Grad

"""
2D version
 input:
# Arguments
 - n1 : number of grid points in the first dimension
 - n2 : number of grid points in the second dimension
 - h1 : distance between grid points in the first dimension
 - h2 : distance between grid points in the second dimension
 - TD_type : type of derivative operator as a string

 output :
 - TD_OP : transform domain operator as a sparse matrix
"""
function get_discrete_Grad(n1,n2,h1::TF,h2::TF,TD_type::String) where {TF<:Real}
    TI=Int64

    #define difference matrix D acting on vectorized model using Kronecker products
    Ix = SparseMatrixCSC{TF}(LinearAlgebra.I,n1,n1) #x
    Iz = SparseMatrixCSC{TF}(LinearAlgebra.I,n2,n2) #z
    Dx = spdiagm(0 => ones(TF,n1-1)*-1, 1 => ones(TF,n1-1)*1); Dx = Dx[1:end-1,:] ./ h1
    Dz = spdiagm(0 => ones(TF,n2-1)*-1, 1 => ones(TF,n2-1)*1); Dz = Dz[1:end-1,:] ./ h2


    if TD_type=="D_z"
      D_OP = kron(Dz,Ix) #D2z
    elseif TD_type=="D_x"
      D_OP = kron(Iz,Dx) #D2x
    elseif  TD_type=="TV" || TD_type=="D2D"
      D2z  = kron(Dz,Ix)
      D2x  = kron(Iz,Dx)
      D_OP = vcat(D2z,D2x) #D2D
    end
      
    return D_OP
end


"""
  3D version
  input: n1 : number of grid points in the first dimension
         n2 : number of grid points in the second dimension
         n3 : number of grid points in the third dimension
         h1 : distance between grid points in the first dimension
         h2 : distance between grid points in the second dimension
         h3 : distance between grid points in the third dimension
         TD_type : type of derivative operator as a string
   output : TD_OP : transform domain operator as a sparse matrix
"""
function get_discrete_Grad(n1,n2,n3,h1::TF,h2::TF,h3::TF,TD_type::String) where {TF<:Real}
    TI = Int64

    #define difference matrix D acting on vectorized model using Kronecker products
    Ix = SparseMatrixCSC{TF}(LinearAlgebra.I,n1,n1) #x
    Iy = SparseMatrixCSC{TF}(LinearAlgebra.I,n2,n2) #x
    Iz = SparseMatrixCSC{TF}(LinearAlgebra.I,n3,n3) #z
    Dx = spdiagm(0 => ones(TF,n1-1)*-1, 1 => ones(TF,n1-1)*1); Dx = Dx[1:end-1,:] ./ h1
    Dy = spdiagm(0 => ones(TF,n2-1)*-1, 1 => ones(TF,n2-1)*1); Dy = Dy[1:end-1,:] ./ h2
    Dz = spdiagm(0 => ones(TF,n3-1)*-1, 1 => ones(TF,n3-1)*1); Dz = Dz[1:end-1,:] ./ h3

    if TD_type=="D_z"
      D_OP = kron(Dz,Iy,Ix)
    elseif TD_type=="D_y"
      D_OP = kron(Iz,Dy,Ix)
    elseif TD_type=="D_x"
      D_OP = kron(Iz,Iy,Dx)
    elseif TD_type=="TV" || TD_type=="D3D"
      D3z = kron(Dz,Iy,Ix)
      D3y = kron(Iz,Dy,Ix)
      D3x = kron(Iz,Iy,Dx)
      D_OP = vcat(D3z,D3y,D3x)
    end

    return D_OP
end
