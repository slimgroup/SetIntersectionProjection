export project_l1_Duchi!

"""
    project_l1_Duchi!(v::Union{Vector{TF},Vector{Complex{TF}}}, b::TF)
  
Projects ector onto L1 ball of specified radius.

w = ProjectOntoL1Ball(v, b) returns the vector w which is the solution
  to the following constrained minimization problem:

   min   ||w - v||_2
   s.t.  ||w||_1 <= b.

  That is, performs Euclidean projection of v to the 1-norm ball of radius
  b.

Author: John Duchi (jduchi@cs.berkeley.edu)
Translated (with some modification) to Julia 1.1 by Bas Peters
"""
function project_l1_Duchi!(v::Union{Vector{TF},Vector{Complex{TF}}}, b::TF) where {TF<:Real}
  b <= TF(0) && error("Radius of L1 ball is negative")
  norm(v, 1) <= b && return v

  lv = length(v)
  u = Vector{TF}(undef,lv)
  sv = Vector{TF}(undef,lv)

  #use RadixSort for Float32 (short keywords)
  copyto!(u, v)
  if TF==Float32
    u = sort!(abs.(u), rev=true,alg=RadixSort)
  else
    u = sort!(abs.(u), rev=true,alg=QuickSort)
  end

  cumsum!(sv, u)

  # Thresholding level
  rho = max(1, min(lv, findlast(u .> ((sv.-b)./ (1.0:1.0:lv)))))
  theta = max.(TF(0) , (sv[rho] .- b) ./ rho)::TF

  # Projection as soft thresholding
  v .= sign.(v) .* max.(abs.(v) .- theta, TF(0))

  return v
end