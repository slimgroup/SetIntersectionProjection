export project_bounds!
"""
Computes the projection of x onto the set of constraints LB <= x <= UB
x may be a vector or a scalar. Uses some manual tricks, attempt to be faster than native julia: x = max.(LB,min.(x,UB))
"""

function project_bounds!{TF<:Real}(x::Vector{TF},LB::TF,UB::TF)
  """
  Computes the projection of x onto the set of constraints LB <= x <= UB
  x is a vector but the bounds are scalars. Uses some manual tricks like @inbounds and multi threading, attempt to be faster than native julia: x. = max.(LB,min.(x,UB))
  """
  Threads.@threads for j in (1:length(x))
    @inbounds x[j] = max(LB,min(x[j],UB))
  end
    return x
end

function project_bounds!{TF<:Real}(x::Vector{TF},LB::Vector{TF},UB::Vector{TF})
  """
  Computes the projection of x onto the set of constraints LB <= x <= UB
  x is a vector and the bounds are vectors as well. Uses some manual tricks like @inbounds and multi threading, attempt to be faster than native julia: x. = max.(LB,min.(x,UB))
  """
@inbounds Threads.@threads for j in (1:length(x))
    #x[i] = max.(LB,min.(x[i],UB))
  @inbounds  x[j] = min(x[j],UB[j])
  @inbounds  x[j] = max(LB[j],x[j])
  end
    return x
end
