# Example 0: Solve 1D Burgers Equation
# u(x,t)_t+(0.5*u²(x,t))_{x}=0
# u(0,x) = f0(x)

using ConservationLawsDiffEq
using OrdinaryDiffEq
using LinearAlgebra

const CFL = 0.5
# First define the problem data (Jacobian is optional but useful)
Jf(u) = u           #Jacobian
f(u) = u^2/2        #Flux function
f0(x) = sin(2*π*x)  #Initial data distribution

# Now discretizate the domain
mesh = Uniform1DFVMesh(10, [0.0, 1.0])

# Now get a explicit semidiscretization (discrete in space) du_h(t)/dt = f_h(u_h(t))
f_h = getSemiDiscretization(f,LaxFriedrichsScheme(),mesh,[Periodic()]; Df = Jf, use_threads = false,numvars = 1)

#Compute discrete initial data
u0 = getInitialState(mesh,f0,use_threads = true)

#Setup ODE problem for a time interval = [0.0,1.0]
ode_prob = ODEProblem(f_h,u0,(0.0,1.0))

#Setup callback in order to fix CFL constant value
cb = getCFLCallback(f_h, CFL)

#Estimate an initial dt
dt = update_dt!(u0, f_h, CFL)

#Solve problem using OrdinaryDiffEq
sol = solve(ode_prob,SSPRK22(); dt = dt, callback = cb)

#Plot solution
#Wrap solution so we can plot using dispatch
u_h = fv_solution(sol, mesh)

using Plots
plot(u_h,tidx = 1,lab="uo",line=(:dot,2)) #Plot inital data
plot!(u_h,lab="LF")                       #Plot LaxFriedrichsScheme solution
