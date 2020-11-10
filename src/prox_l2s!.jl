export prox_l2s!

function prox_l2s!(x::Vector{TF}, rho::TF, m::Vector{TF}) where {TF<:Real}
  x .= (x .* rho .+ m) ./ (rho .+ 1.0);
  return x
end
