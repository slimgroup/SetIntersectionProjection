#compare parallel Dykstra vs PARSDMM
#look at the total number of PARDMM iterations if
#we use ARADMM to solve Dykstra subproblems.

@everywhere using SetIntersectionProjection
include(joinpath(Pkg.dir("SetIntersectionProjection"),"examples/Dykstra_prox_parallel.jl"))
include(joinpath(Pkg.dir("SetIntersectionProjection"),"examples/Dykstra_prox_parallel2.jl"))
using MAT
using PyPlot

type compgrid
  d :: Tuple
  n :: Tuple
end

data_dir = "/Volumes/Users/bpeters/Downloads"#/data/slim/bpeters/SetIntersection_data_results"
file = matopen(joinpath(data_dir,"compass_velocity.mat"))
m=read(file, "Data")
close(file)
m=m[1:341,200:600];
m=m';

#PARSDMM options:
options=PARSDMM_options()
options.FL=Float32
options.feas_tol=0.01f0
#options=default_PARSDMM_options(options,options.FL)
set_zero_subnormals(true)

#select working precision
if options.FL==Float64
  TF = Float64
  TI = Int64
elseif options.FL==Float32
  TF = Float32
  TI = Int32
end

comp_grid = compgrid((TF(25), TF(6)),(size(m,1), size(m,2)))
m=convert(Vector{TF},vec(m))
m2=deepcopy(m)
m3=deepcopy(m)

#constraints (bounds, vertical monotonicity, total-variation)
constraint=Dict()

constraint["use_bounds"]=true
constraint["m_min"]=1500
constraint["m_max"]=4000

constraint["use_TD_bounds_1"]=true;
constraint["TDB_operator_1"]="D_z";
constraint["TD_LB_1"]=0;
constraint["TD_UB_1"]=1e6;

constraint["use_TD_l1_1"]      = true
constraint["TD_l1_operator_1"] = "TV"
(TV_OP, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"TV",TF)
constraint["TD_l1_sigma_1"]    = 0.1*norm(TV_OP*m,1)


BLAS.set_num_threads(2) #2 is fine for a small problem
(P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL)
(TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(TD_OP,TD_Prop,comp_grid,options)

println("PARSDMM serial (bounds, bounds on D_z and TV):")
@time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);

cg_axis=[0 ; cumsum(log_PARSDMM.cg_it)];
cg_axis=cg_axis[1:10:end]

feas_axis=0:10:(10-1)*size(log_PARSDMM.set_feasibility,1)

#define axis limits and colorbar limits
xmax = comp_grid.d[1]*comp_grid.n[1]
zmax = comp_grid.d[2]*comp_grid.n[2]
vmi=1500
vma=4500
figure();imshow(reshape(m,(comp_grid.n[1],comp_grid.n[2]))',cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("model to project")
figure();imshow(reshape(x,(comp_grid.n[1],comp_grid.n[2]))',cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("Projection PARSDMM (bounds, bounds on D_z, TV))")


# parallel Dykstra: first set up projectors onto complicated sets
#use same stopping conditions as for PARSDMM
maxit_dyk=200
dyk_feas_tol = deepcopy(options.feas_tol)
obj_dyk_tol  = deepcopy(options.obj_tol)

feas_tol_target= deepcopy(options.feas_tol)
obj_tol_target = deepcopy(options.obj_tol)

#use more accurate ARADMM sub-problem solutions
# we can play with these a bit to see if we can decrease total computational cost
# if set too loose, we will not see overall convergence
options.obj_tol=0.8*options.obj_tol
options.feas_tol=0.8*options.feas_tol
options.evol_rel_tol=10*eps(TF)

P=Vector{Any}(3)
P[1]=P_sub[1] #projector onto bounds is easy.

# Set up ARADMM to project onto transform-domain bounds (PARSDMM with 1 constraint set is equivalent to ARADMM)
constraint=Dict()
constraint["use_TD_bounds_1"]=true;
constraint["TDB_operator_1"]="D_z";
constraint["TD_LB_1"]=0;
constraint["TD_UB_1"]=1e6;
(P_sub2,TD_OP2,TD_Prop2) = setup_constraints(constraint,comp_grid,options.FL)
(TD_OP2,AtA2,l,y) = PARSDMM_precompute_distribute(TD_OP2,TD_Prop2,comp_grid,options)
P[2] = inp -> PARSDMM(inp,AtA2,TD_OP2,TD_Prop2,P_sub2,comp_grid,options) #projector onto transform-domain bounds

# Set up ARADMM to project onto anisotropic-TV set (PARSDMM with 1 constraint set is equivalent to ARADMM)
constraint=Dict()
constraint["use_TD_l1_1"]      = true
constraint["TD_l1_operator_1"] = "TV"
(TV_OP, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"TV",TF)
constraint["TD_l1_sigma_1"]    = 0.1*norm(TV_OP*m,1)
(P_sub3,TD_OP3,TD_Prop3) = setup_constraints(constraint,comp_grid,options.FL)
(TD_OP3,AtA3,l,y) = PARSDMM_precompute_distribute(TD_OP3,TD_Prop3,comp_grid,options)
P[3] = inp -> PARSDMM(inp,AtA3,TD_OP3,TD_Prop3,P_sub3,comp_grid,options) #projector onto transform-domain bounds

closed_form=Vector{Bool}(3)
closed_form[1]=true
closed_form[2]=false
closed_form[3]=false

figure();imshow(reshape(m2,(comp_grid.n[1],comp_grid.n[2]))',cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("model to project")

@time (x,obj,feasibility_err_dyk,cg_it,ARADMM_it,l1_P,bounds_P)=Dykstra_prox_parallel(m2,P,P_sub,TD_OP,closed_form,maxit_dyk,dyk_feas_tol,obj_dyk_tol)

figure();imshow(reshape(x,(comp_grid.n[1],comp_grid.n[2]))',cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("Projection Dykstra (bounds, bounds on D_z, TV)")


fig, ax = subplots()
ax[:semilogy](feas_axis,log_PARSDMM.set_feasibility[:,1],color="b",label="PARSDMM - bounds");
ax[:semilogy](feas_axis,log_PARSDMM.set_feasibility[:,2],color="r",label="PARSDMM - monotonicity");
ax[:semilogy](feas_axis,log_PARSDMM.set_feasibility[:,3],color="k",label="PARSDMM - TV");
ax[:semilogy]([0 ; cumsum(ARADMM_it)],feasibility_err_dyk[:,1],linestyle="--",color="b",label="Parallel Dykstra - bounds");
ax[:semilogy]([0 ; cumsum(ARADMM_it)],feasibility_err_dyk[:,2],linestyle="--",color="r",label="Parallel Dykstra - monotonicity");
ax[:semilogy]([0 ; cumsum(ARADMM_it)],feasibility_err_dyk[:,3],linestyle="--",color="k",label="Parallel Dykstra - TV");
# ax[:loglog](feas_axis,log_PARSDMM.set_feasibility[:,3],color="k",label="PARSDMM - TV");
# ax[:loglog](feas_axis,log_PARSDMM.set_feasibility[:,1],color="b",label="PARSDMM - bounds");
# ax[:loglog](feas_axis,log_PARSDMM.set_feasibility[:,2],color="r",label="PARSDMM - monotonicity");
# ax[:loglog]([0 ; cumsum(ARADMM_it)],feasibility_err_dyk[:,1],linestyle="--",color="b",label="Parallel Dykstra - bounds");
# ax[:loglog]([0 ; cumsum(ARADMM_it)],feasibility_err_dyk[:,2],linestyle="--",color="r",label="Parallel Dykstra - monotonicity");
# ax[:loglog]([0 ; cumsum(ARADMM_it)],feasibility_err_dyk[:,3],linestyle="--",color="k",label="Parallel Dykstra - TV");
#ax[:semilogy]([0 ; cumsum(ARADMM_it)],ones(size(feasibility_err_dyk,1)).*feas_tol_target,label="target");
title("relative set feasibility error", fontsize=16)
xlabel(L"number of sequential $\ell_1$ projections", fontsize=16)
ax[:legend](fontsize=12)
ax[:tick_params]("both",labelsize=12)
savefig("Dykstra_vs_PARSDMM_feasibility.eps",bbox_inches="tight",dpi=1200)
savefig("Dykstra_vs_PARSDMM_feasibility.png",bbox_inches="tight")


fig, ax = subplots()
ax[:semilogy](cg_axis,log_PARSDMM.set_feasibility[1:end-1,1],color="b",label="PARSDMM - bounds");
ax[:semilogy](cg_axis,log_PARSDMM.set_feasibility[1:end-1,2],color="r",label="PARSDMM - monotonicity");
ax[:semilogy](cg_axis,log_PARSDMM.set_feasibility[1:end-1,3],color="k",label="PARSDMM - TV");
ax[:semilogy]([0;cumsum(cg_it)],feasibility_err_dyk[:,1],linestyle="--",color="b",label="Parallel Dykstra - bounds");
ax[:semilogy]([0;cumsum(cg_it)],feasibility_err_dyk[:,2],linestyle="--",color="r",label="Parallel Dykstra - monotonicity");
ax[:semilogy]([0;cumsum(cg_it)],feasibility_err_dyk[:,3],linestyle="--",color="k",label="Parallel Dykstra - TV");
# ax[:loglog](cg_axis,log_PARSDMM.set_feasibility[1:end-1,1],color="b",label="PARSDMM - bounds");
# ax[:loglog](cg_axis,log_PARSDMM.set_feasibility[1:end-1,2],color="r",label="PARSDMM - monotonicity");
# ax[:loglog](cg_axis,log_PARSDMM.set_feasibility[1:end-1,3],color="k",label="PARSDMM - TV");
# ax[:loglog]([0;cumsum(cg_it)],feasibility_err_dyk[:,1],linestyle="--",color="b",label="Parallel Dykstra - bounds");
# ax[:loglog]([0;cumsum(cg_it)],feasibility_err_dyk[:,2],linestyle="--",color="r",label="Parallel Dykstra - monotonicity");
# ax[:loglog]([0;cumsum(cg_it)],feasibility_err_dyk[:,3],linestyle="--",color="k",label="Parallel Dykstra - TV");

#ax[:semilogy]([0 ; cumsum(ARADMM_it)],ones(size(feasibility_err_dyk,1)).*feas_tol_target,label="target");
title("relative set feasibility error", fontsize=16)
xlabel("number of sequential CG iterations", fontsize=16)
ax[:legend](fontsize=12)
ax[:tick_params]("both",labelsize=12)
savefig("Dykstra_vs_PARSDMM_feasibility_CG.eps",bbox_inches="tight",dpi=1200)
savefig("Dykstra_vs_PARSDMM_feasibility_CG.png",bbox_inches="tight")


fig, ax = subplots()
ax[:semilogy](abs.(diff(log_PARSDMM.obj)./log_PARSDMM.obj[1:end-1]),color="b",label="PARSDMM");
ax[:semilogy](cumsum(ARADMM_it[2:end]),abs.(diff(obj)./obj[1:end-1]),color="r",label="Parallel Dykstra");
#ax[:semilogy](ones(sum(ARADMM_it[2:end])).*obj_tol_target,label="target");
title(L"relative change in $|| \mathbf{m} - \mathbf{x} ||$", fontsize=16)
xlabel(L"number of sequential $\ell_1$ projections", fontsize=16)
ax[:legend](fontsize=12)
ax[:tick_params]("both",labelsize=12)
savefig("Dykstra_vs_PARSDMM_obj_change.eps",bbox_inches="tight",dpi=1200)
savefig("Dykstra_vs_PARSDMM_obj_change.png",bbox_inches="tight")

#######################################################################################
#######################################################################################

#now with a different set of constraints:
# transform-domain rank and bounds
constraint=Dict()

#bound constraints
constraint["use_bounds"]=true
constraint["m_min"]=1450
constraint["m_max"]=4000

#nuclear norm constraint on vertical derivative of the image
constraint["use_TD_rank_1"]=true
constraint["TD_rank_operator_1"]="D_z"
constraint["TD_max_rank_1"]=15


BLAS.set_num_threads(2) #2 is fine for a small problem
(P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL)
(TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(TD_OP,TD_Prop,comp_grid,options)

println("PARSDMM serial (bounds, bounds on D_z and TV):")
@time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);

cg_axis=[0 ; cumsum(log_PARSDMM.cg_it)];
cg_axis=cg_axis[1:10:end]

feas_axis=0:10:(10-1)*size(log_PARSDMM.set_feasibility,1)

#define axis limits and colorbar limits
xmax = comp_grid.d[1]*comp_grid.n[1]
zmax = comp_grid.d[2]*comp_grid.n[2]
vmi=1500
vma=4500
figure();imshow(reshape(m,(comp_grid.n[1],comp_grid.n[2]))',cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("model to project")
figure();imshow(reshape(x,(comp_grid.n[1],comp_grid.n[2]))',cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("Projection PARSDMM (bounds, bounds on D_z, TV))")

# parallel Dykstra: first set up projectors onto complicated sets
#use same stopping conditions as for PARSDMM
maxit_dyk=12
dyk_feas_tol = deepcopy(options.feas_tol)
obj_dyk_tol  = deepcopy(options.obj_tol)

feas_tol_target= deepcopy(options.feas_tol)
obj_tol_target = deepcopy(options.obj_tol)

#use more accurate ARADMM sub-problem solutions
# we can play with these a bit to see if we can decrease total computational cost
# if set too loose, we will not see overall convergence
options.obj_tol=0.5*options.obj_tol
options.feas_tol=0.5*options.feas_tol
options.evol_rel_tol=10*eps(TF)

P=Vector{Any}(2)
P[1]=P_sub[1] #projector onto bounds is easy.

# Set up ARADMM to project onto set of transform-domain rank (PARSDMM with 1 constraint set is equivalent to ARADMM)
constraint=Dict()
constraint["use_TD_rank_1"]=true;
constraint["TD_rank_operator_1"]="D_z";
constraint["TD_max_rank_1"]=15;
(P_sub2,TD_OP2,TD_Prop2) = setup_constraints(constraint,comp_grid,options.FL)
(TD_OP2,AtA2,l,y) = PARSDMM_precompute_distribute(TD_OP2,TD_Prop2,comp_grid,options)
P[2] = inp -> PARSDMM(inp,AtA2,TD_OP2,TD_Prop2,P_sub2,comp_grid,options) #projector onto set of transform-domain rank

closed_form=Vector{Bool}(2)
closed_form[1]=true
closed_form[2]=false

figure();imshow(reshape(m2,(comp_grid.n[1],comp_grid.n[2]))',cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("model to project")

@time (x,obj,feasibility_err_dyk,cg_it,ARADMM_it,svd_P)=Dykstra_prox_parallel(m2,P,P_sub,TD_OP,closed_form,maxit_dyk,dyk_feas_tol,obj_dyk_tol)

figure();imshow(reshape(x,(comp_grid.n[1],comp_grid.n[2]))',cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("Projection Dykstra (bounds, bounds on D_z, TV)")

fig, ax = subplots()
ax[:semilogy](feas_axis,log_PARSDMM.set_feasibility[:,1],color="b",label="PARSDMM - bounds");
ax[:semilogy](feas_axis,log_PARSDMM.set_feasibility[:,2],color="r",label="PARSDMM - TD rank");
ax[:semilogy]([0 ; cumsum(ARADMM_it)],feasibility_err_dyk[:,1],linestyle="--",color="b",label="Parallel Dykstra - bounds");
ax[:semilogy]([0 ; cumsum(ARADMM_it)],feasibility_err_dyk[:,2],linestyle="--",color="r",label="Parallel Dykstra - TD rank");
#ax[:semilogy]([0 ; cumsum(ARADMM_it)],ones(size(feasibility_err_dyk,1)).*feas_tol_target,label="target");
title("relative set feasibility error", fontsize=16)
xlabel("number of sequential SVDs", fontsize=16)
ax[:legend](fontsize=12)
ax[:tick_params]("both",labelsize=12)
savefig("Dykstra_vs_PARSDMM_feasibility2.eps",bbox_inches="tight",dpi=1200)
savefig("Dykstra_vs_PARSDMM_feasibility2.png",bbox_inches="tight")


fig, ax = subplots()
ax[:semilogy](cg_axis,log_PARSDMM.set_feasibility[1:end-1,1],color="b",label="PARSDMM - bounds");
ax[:semilogy](cg_axis,log_PARSDMM.set_feasibility[1:end-1,2],color="r",label="PARSDMM - TD rank");
ax[:semilogy]([0;cumsum(cg_it)],feasibility_err_dyk[:,1],linestyle="--",color="b",label="Parallel Dykstra - bounds");
ax[:semilogy]([0;cumsum(cg_it)],feasibility_err_dyk[:,2],linestyle="--",color="r",label="Parallel Dykstra - TD rank");
#ax[:semilogy]([0 ; cumsum(ARADMM_it)],ones(size(feasibility_err_dyk,1)).*feas_tol_target,label="target");
title("relative set feasibility error", fontsize=16)
xlabel("number of sequential CG iterations", fontsize=16)
ax[:legend]()
ax[:tick_params]("both",labelsize=12)
savefig("Dykstra_vs_PARSDMM_feasibility_CG2.eps",bbox_inches="tight",dpi=1200)
savefig("Dykstra_vs_PARSDMM_feasibility_CG2.png",bbox_inches="tight")


fig, ax = subplots()
ax[:semilogy](abs.(diff(log_PARSDMM.obj)./log_PARSDMM.obj[1:end-1]),color="b",label="PARSDMM");
ax[:semilogy](cumsum(ARADMM_it[2:end]),abs.(diff(obj)./obj[1:end-1]),color="r",label="Parallel Dykstra");
#ax[:semilogy](ones(sum(ARADMM_it[2:end])).*obj_tol_target,label="target");
title(L"relative change in $|| \mathbf{m} - \mathbf{x} ||$", fontsize=16);
xlabel("number of sequential SVDs", fontsize=16)
ax[:legend](fontsize=12)
ax[:tick_params]("both",labelsize=12) 
savefig("Dykstra_vs_PARSDMM_obj_change2.eps",bbox_inches="tight",dpi=1200)
savefig("Dykstra_vs_PARSDMM_obj_change2.png",bbox_inches="tight")
