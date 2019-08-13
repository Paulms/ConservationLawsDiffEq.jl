using DifferentialEquations
using Tensors
using StaticArrays
import ForwardDiff
import FastGaussQuadrature
include("mesh.jl")
include("mesh_generator.jl")
include("fvmesh.jl")
include("fvflux.jl")
include("fvintegrator.jl")
include("fvalgorithmsAPI.jl")
include("timecallbacks.jl")
include("fvutils.jl")
include("fvsolve.jl")

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

mesh = line_mesh(LineCell, 10, 0.0, 1.0)
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
u0 = getInitialState(mesh,f0,method=:average)
tspan = (0.0,1.0)
prob = ODEProblem(f_sd,u0,tspan)
f_sd.dt = 0.1
sol = solve(prob, SSPRK22(), dt = f_sd.dt)

cb = get_adaptative_callback(f_sd)
sol = solve(prob, callback = cb)

Juno.Profile.clear()
Juno.@profile solve(prob, SSPRK22(), dt = f_sd.dt)
Juno.profiler()

CFL = 0.5
f_sd = getSemiDiscretization(f,Upwind(),mesh,[dbc], CFL=CFL)
cb = getCFLCallback(f_sd)

using BenchmarkTools
@btime solve(prob, SSPRK22(), dt = f_sd.dt)
f_sd.dt = update_dt!(u0, f_sd)
@btime solve(prob, SSPRK22(), dt = f_sd.dt, callback = cb)

using Plots
xn = get_nodes_matrix(mesh)
x = [xn[i] + 0.5*(xn[i+1]-xn[i]) for i in 1:getncells(mesh)]
plot(x, sol.u[end])
