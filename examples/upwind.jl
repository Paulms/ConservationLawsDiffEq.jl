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

function update_cell_value(uold,cell_idx,dt,mesh,Flux,alg::Upwind)
    # Local speeds of propagation
    @inbounds ul=cellval_at_left(cell_idx,uold,mesh)
    @inbounds ur=cellval_at_right(cell_idx,uold,mesh)
    um = uold[cell_idx]
    dx = cell_volume(mesh,j)
    a = evalJacobian(Flux,um)
    a_plus = max(a,0)
    a_minus = min(a,0)
    # Numerical Fluxes
    return a_plus*(Flux(um) - Flux(ul))/dx + a_minus*(Flux(ur) - Flux(um))/dx
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

function compute_fluxes!(fluxes, Flux, u, mesh, dt, alg::AbstractFVAlgorithm, ::Type{Val{false}})
    #update vector
    for j in node_indices(mesh)
        fluxes[:,j] .= update_cell_value(u,j,dt,mesh,Flux,alg)
    end
end

function compute_du!(du, fluxes, mesh, ::Type{Val{false}})
    for cell in cell_indices(mesh)
        @inbounds left = left_node_idx(cell, mesh)
        @inbounds right = right_node_idx(cell, mesh)
        @inbounds du[:,cell] = -(fluxes[:,right] - fluxes[:,left] ) / cell_volume(cell, mesh)
    end
end

function (fv::FVIntegrator)(du::AbstractArray{T,2}, u::AbstractArray{T,2}, p, t) where {T}
  mesh = fv.mesh; alg = fv.alg; Flux = fv.Flux; M = fv.M;
  fluxes = fv.fluxes; dt = fv.dt; use_threads = fv.use_threads
  compute_fluxes!(fluxes, Flux, u, mesh, dt, alg, Val{use_threads})
  if isleftzeroflux(mesh);fluxes[:,1] .= zero(T); end
  if isrightzeroflux(mesh);fluxes[:,getnnodes(mesh)] .= zero(T);end
  compute_du!(du, fluxes, mesh, Val{use_threads})
  if isleftdirichlet(mesh);du[:,1] .= zero(T); end
  if isrightdirichlet(mesh);du[:,getnnodes(mesh)] .= zero(T);end
  nothing
end

function getSemiDiscretization(f,alg::AbstractFVAlgorithm,
    mesh,dbcs; Df = nothing, numvars = 1, CFL = nothing,
    Type = Float64, use_threads = false)
    internal_mesh = mesh_setup(mesh, dbcs)
    fluxes = MMatrix{numvars,getnnodes(mesh),Type}(undef)
    dt = 0.0
    Flux = flux_function(f, Df)
    FVIntegrator(alg,mesh,Flux,CFL,numvars, fluxes, dt, use_threads)
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
        u0[:,i] = num_integrate(f0,nodes[i], nodes[i+1])/cell_volume(mesh,i)
    elseif method == :eval_in_centers
        u0[:,i] = f0((nodes[i]+nodes[i+1])/2)
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
    return u0
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
# Define initial conditions
f0(x) = sin(2*π*x)

# Use DifferentialEquations.jl to solve in time
u0=getInitialState(mesh,f0,method=:average)
tspan = (0.0,1.0)
prob = ODEProblem(f_sd,u0,tspan)
sol = solve(prob)
