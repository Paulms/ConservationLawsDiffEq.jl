using DifferentialEquations
using Tensors
using StaticArrays
import ForwardDiff
import FastGaussQuadrature
include("mesh.jl")
include("mesh_generator.jl")
include("fvmesh.jl")
include("fvflux.jl")

abstract type AbstractFVAlgorithm end

struct Upwind <: AbstractFVAlgorithm; end

function update_flux_value(uold,node_idx,dt,dx,mesh,Flux,alg::Upwind)
    # Local speeds of propagation
    @inbounds ul=cellval_at_left(node_idx,uold,mesh)
    @inbounds ur=cellval_at_right(node_idx,uold,mesh)
    #a = evalJacobian(Flux,0.5(ul+ur))
    #a_plus = a > 0
    #a_minus = a < 0
    # Numerical Fluxes
    return 0.5*(Flux(ul)+Flux(ur))-dx/(2*dt)*(ur-ul) #a_plus*Flux(ul) + a_minus*Flux(ur)
end

mutable struct FVIntegrator{T1,mType,F,cType,T2,tType}
  alg::T1
  mesh::mType
  Flux::F
  CFL :: cType
  numvars::Int
  fluxes :: T2
  dt :: tType
  use_threads :: Bool
end

function compute_fluxes!(fluxes, Flux, u, mesh, dt, alg::AbstractFVAlgorithm, noscalar::Bool, ::Type{Val{false}})
    # This works assuming uniform mesh
    dx = cell_volume(mesh, 1)
    #update vector
    for j in node_indices(mesh)
        if noscalar
            fluxes[:,j] .= update_flux_value(u,j,dt,dx,mesh,Flux,alg)
        else
            fluxes[j] = update_flux_value(u,j,dt,dx,mesh,Flux,alg)
        end
    end
end

function compute_du!(du, fluxes, mesh, noscalar::Bool, ::Type{Val{false}})
    for cell in cell_indices(mesh)
        @inbounds left = left_node_idx(cell, mesh)
        @inbounds right = right_node_idx(cell, mesh)
        if noscalar
            du[:,cell] = -(fluxes[:,right] - fluxes[:,left] ) / cell_volume(mesh.mesh, cell)
        else
            du[cell] = -(fluxes[right] - fluxes[left] ) / cell_volume(mesh.mesh, cell)
        end
    end
end

function (fv::FVIntegrator)(du::AbstractArray{T}, u::AbstractArray{T}, p, t) where {T}
  mesh = fv.mesh; alg = fv.alg; Flux = fv.Flux; numvars = fv.numvars
  fluxes = fv.fluxes; dt = fv.dt; use_threads = fv.use_threads
  compute_fluxes!(fluxes, Flux, u, mesh, dt, alg, numvars > 1, Val{use_threads})
  apply_bc_in_fluxes!(fluxes, mesh)
  compute_du!(du, fluxes, mesh, numvars > 1, Val{use_threads})
  apply_bc_in_du!(du, mesh)
  nothing
end

function getSemiDiscretization(f,alg::AbstractFVAlgorithm,
    mesh,dbcs; Df = nothing, numvars = 1, CFL = nothing,
    Type = Float64, use_threads = false)
    probtype = numvars > 1 ? GeneralProblem() : ScalarProblem()
    internal_mesh = mesh_setup(mesh, dbcs,probtype)
    fluxes = numvars > 1 ? MMatrix{numvars,getnnodes(mesh),Type}(undef) : MVector{getnnodes(mesh),Type}(undef)
    dt = 0.0
    Flux = numvars > 1 ? flux_function(f, Df) : scalar_flux_function(f, Df)
    FVIntegrator(alg,internal_mesh,Flux,CFL,numvars, fluxes, dt, use_threads)
end

function num_integrate(f,a,b;order=5, method = FastGaussQuadrature.gausslegendre)
    nodes, weights = method(order);
    t_nodes = 0.5*(b-a)*nodes .+ 0.5*(b+a)
    M = length(f(a))
    tmp = fill(0.0,M)
    for i in 1:M
        g(x) = f(x)[i]
        tmp[i] = 0.5*(b-a)*dot(g.(t_nodes),weights)
    end
    return tmp
end

function initial_data_inner_loop!(u0, f0, nodes, method, mesh, i)
    if method == :average
        u0[:,i] .= num_integrate(f0,nodes[i], nodes[i+1])/cell_volume(mesh,i)
    elseif method == :eval_in_centers
        u0[:,i] .= f0((nodes[i]+nodes[i+1])/2)
    else
        error("invalid initial data processing method: ", method)
    end
    nothing
end

function getInitialState(mesh, f0; method =:average, use_threads = false, MType = Float64)
    N = getncells(mesh)
    numvars = size(f0(getnodecoords(mesh, 1)[1]),1)
    u0 = MMatrix{numvars,N,MType}(undef)
    nodes = get_nodes_matrix(mesh)
    if use_threads
        Threads.@threads for i in 1:getncells(mesh)
            initial_data_inner_loop!(u0, f0, nodes, method, mesh, i)
        end
    else
        for i in 1:getncells(mesh)
            initial_data_inner_loop!(u0, f0, nodes, method, mesh, i)
        end
    end
    return numvars > 1 ? u0 : u0[1,:]
end

mesh = line_mesh(LineCell, (10,), Tensors.Vec{1}((0.0,)), Tensors.Vec{1}((1.0,)))
#seup problem
# 1D Burgers Equation
# u_t=-(0.5*u²)_{x}
f(u) = -u^2/2
# ### Boundary conditions
dbc = Periodic()
# Discretize in space
f_sd = getSemiDiscretization(f,Upwind(),mesh,[dbc])

struct AdaptFVCallbackAffect{F}
    f_sd::F
end
function (f::AdaptFVCallbackAffect)(integrator)
    f_sd.dt = get_proposed_dt(integrator)
end

function get_adaptative_callback(f_sd)
    condition = (u,t,integrator) -> true
    affect! = AdaptFVCallbackAffect(f_sd)
    return DiscreteCallback(condition,affect!,save_positions=(false, false))
end
cb = get_adaptative_callback(f_sd)

# Define initial conditions
f0(x) = sin(2*π*x)

# Use DifferentialEquations.jl to solve in time
u0 = getInitialState(mesh,f0,method=:average)
tspan = (0.0,1.0)
prob = ODEProblem(f_sd,u0,tspan)
f_sd.dt = 0.1
sol = solve(prob, SSPRK22(), dt = f_sd.dt)
sol = solve(prob, callback = cb)

Juno.Profile.clear()
Juno.@profile solve(prob, SSPRK22(), dt = f_sd.dt)
Juno.profiler()

CFL = 0.5
f_sd = getSemiDiscretization(f,Upwind(),mesh,[dbc], CFL=CFL)

function update_dt(alg::AbstractFVAlgorithm,u,Flux,
    CFL,mesh)
  maxρ = zero(eltype(u))
  dx = cell_volume(mesh, 1)
  for i in cell_indices(mesh)
    maxρ = max(maxρ, fluxρ(value_at_cell(u,i,mesh), Flux))
  end
  CFL/(1/dx*maxρ)
end
function update_dt!(u,fv::FVIntegrator)
  fv.dt = update_dt(fv.alg, u, fv.Flux, fv.CFL, fv.mesh)
  fv.dt
end

function getCFLCallback(f_sd)
    dtFE(u,p,t) = update_dt!(u, f_sd)
    StepsizeLimiter(dtFE;safety_factor=1.0,max_step=true,cached_dtcache=0.0)
end
cb = getCFLCallback(f_sd)

using BenchmarkTools
f_sd.dt = update_dt!(u0, f_sd)
@btime solve(prob, SSPRK22(), dt = f_sd.dt, callback = cb)

using Plots
xn = get_nodes_matrix(mesh)
x = [xn[i] + 0.5*(xn[i+1]-xn[i]) for i in 1:getncells(mesh)]
plot(x, sol.u[end])
