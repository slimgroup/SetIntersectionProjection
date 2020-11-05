module SetIntersectionProjection

#packages that used to be in base in Julia 0.6
using Distributed
using LinearAlgebra
using SparseArrays
using Printf
using FFTW
using Statistics

#other packages
using Parameters
using Interpolations
using DistributedArrays
using JOLI
using Wavelets
using SortingAlgorithms

export log_type_PARSDMM, set_properties, PARSDMM_options, set_definitions

#main scripts
include("PARSDMM.jl")
include("PARSDMM_multi_level.jl")
include("PARSDMM_precompute_distribute.jl")

#algorithms and solver scripts
include("argmin_x.jl");
include("adapt_rho_gamma.jl");
include("update_y_l.jl");
include("stop_PARSDMM.jl");
include("PARSDMM_initialize.jl")
include("Q_update!.jl")
include("rhs_compose.jl")

#linear algebra functions
include("cg.jl");
include("mat2CDS.jl")
include("CDS_MVp.jl")
include("CDS_MVp_MT.jl")
include("CDS_MVp_MT_subfunc.jl")
include("CDS_scaled_add!.jl")

#scripts for parallelism
include("update_y_l_parallel.jl")
include("adapt_rho_gamma_parallel.jl")
include("compute_relative_feasibility.jl")

#multi-threaded scripts
include("a_is_b_min_c_MT!.jl")

#multi level related scripts
include("setup_multi_level_PARSDMM.jl")
include("constraint2coarse.jl")
include("interpolate_y_l.jl")

#scripts for setting up constraints, projetors, linear operators
include("default_PARSDMM_options.jl")
include("convert_options!.jl")
include("get_discrete_Grad.jl");
include("get_TD_operator.jl");
include("get_projector.jl");
include("get_bound_constraints.jl");
include("setup_constraints.jl");

#projectors
include("projectors/project_bounds!.jl")
include("projectors/project_cardinality!.jl");
include("projectors/project_rank!.jl");
include("projectors/project_nuclear!.jl");
include("projectors/project_l1_Duchi!.jl");
include("projectors/project_l2!.jl");
include("projectors/project_annulus!.jl");
include("projectors/project_subspace!.jl");
include("projectors/project_histogram_relaxed.jl");

#other proximal maps
include("prox_l2s!.jl")
include("prox_l1!.jl")

#scripts that are required to run examples
include("constraint_learning_by_observation.jl")

#Minkowski sum related scripts
include("PARSDMM_precompute_distribute_Minkowski.jl")

#define types
mutable struct log_type_PARSDMM
      set_feasibility   :: Array{Real,2}
      r_dual            :: Array{Real,2}
      r_pri             :: Array{Real,2}
      r_dual_total      :: Vector{Real}
      r_pri_total       :: Vector{Real}
      obj               :: Vector{Real}
      evol_x            :: Vector{Real}
      rho               :: Array{Real,2}
      gamma             :: Array{Real,2}
      cg_it             :: Vector{Integer}
      cg_relres         :: Vector{Real}
      T_cg              :: Real
      T_stop            :: Real
      T_ini             :: Real
      T_rhs             :: Real
      T_adjust_rho_gamma:: Real
      T_y_l_upd         :: Real
      T_Q_upd           :: Real
end

@with_kw mutable struct PARSDMM_options
  x_min_solver          :: String       = "CG_normal" #what algorithm to use for the x-minimization (CG applied to normal equations)
  maxit                 :: Integer      = 200         #max number of PARSDMM iterations
  evol_rel_tol          :: Real         = 1e-3        #stop PARSDMM if ||x^k - X^{k-1}||_2 / || x^k || < options.evol_rel_tol AND options.feas_tol is reached
  feas_tol              :: Real         = 5e-2        #stop PARSDMM if the transform-domain relative feasibility error is < options.feas_tol AND options.evol_rel_tol is reached
  obj_tol               :: Real         = 1e-3        #optional stopping criterion for change in distance from point that we want to project
  rho_ini               :: Vector{Real} = [10.0]      #initial values for the augmented-Lagrangian penalty parameters. One value in array or one value per constraint set in array
  rho_update_frequency  :: Integer      = 2           #update augmented-Lagrangian penalty parameters and relaxation parameters every X number of PARSDMM iterations
  gamma_ini             :: Real         = 1.0         #initial value for all relaxation parameters (scalar)
  adjust_rho            :: Bool         = true        #adapt augmented-Lagrangian penalty parameters or not
  adjust_gamma          :: Bool         = true        #adapt relaxation parameters in PARSDMM
  adjust_feasibility_rho:: Bool         = true        #adapt augmented-Lagrangian penalty parameters based on constraint set feasibility errors (can be used in combination with options.adjust_rho)
  Blas_active           :: Bool         = true        #use direct BLAS calls, otherwise the code will use Julia loop-fusion where possible
  feasibility_only      :: Bool         = false       #drop distance term and solve a feasibility problem
  FL                    :: DataType     = Float32     #type of Float: Float32 or Float64
  parallel              :: Bool         = false       #compute proximal mappings, multiplier updates, rho and gamma updates in parallel
  zero_ini_guess        :: Bool         = true        #zero initial guess for primal, auxiliary, and multipliers
  Minkowski             :: Bool         = false       #the intersection of sets includes a Minkowski set
end

#save properties of the constraint set and its linear operator (not all are currently used in the code)
#each entry in the following vectors contain the information corresponding to one set
mutable struct set_properties
           ncvx       ::Vector{Bool}                                          #is the set non-convex? (true/false)
           AtA_diag   ::Vector{Bool}                                          #is A^T A a diagonal matrix?
           dense      ::Vector{Bool}                                          #is A a dense matrix?
           TD_n       ::Vector{Tuple}                                         #the grid dimensions in the transform-domain.
           tag        ::Vector{Tuple{String,String,String,String}}            #(constraint type, linear operator description, application mode,application direction if applicable)
           banded     ::Vector{Bool}                                          #is A a banded matrix?
           AtA_offsets::Union{Vector{Vector{Int32}},Vector{Vector{Int64}}}    #only required if A is banded. A vector of indices of the non-zero diagonals, where the main diagonal is index 0
end

mutable struct set_definitions
  set_type          ::String #"bounds","l1","l2","annulus","subspace","nuclear","rank","histogram","cardinality"
  TD_OP             ::String
  min               ::Union{Vector,Real}
  max               ::Union{Vector,Real}
  app_mode          ::Tuple{String,String} #("tensor/"matrix"/"slice"/"fiber" , "x","y","z") , x,y,z apply only to slice and fibers of a tensor or matrix
  custom_TD_OP      ::Tuple{Union{Array{Any},Array{Float64,2},Array{Float32,2}, joAbstractLinearOperator},Bool} #custom matrix for subspace, and boolean to indicate orthogonality
end

end # module
