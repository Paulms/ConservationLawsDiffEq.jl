using ConservationLawsDiffEq
using OrdinaryDiffEq
using LinearAlgebra
u_exact(x,t) = sin(4*pi*(x-t))
# Define the flux and flux Jacobian
Jf(u) = 1
f(u) = u

# Initial condition (using integral cell averages)
f0(x) = sin(4*pi*x)

#Setup problem for a given N (number of cells/control volumenes on a uniform mesh)
#and given final time (Tend) with periodic boundary conditions
function get_problem(N, scheme; Tend = 2.0, CFL = 0.5)

  mesh = Uniform1DFVMesh(N, [0.0, 1.0])

  f_h = getSemiDiscretization(f,scheme,mesh,[Periodic()]; Df = Jf, use_threads = false,numvars = 1)

  #Compute discrete initial data
  u0 = getInitialState(mesh,f0,use_threads = true)

  #Setup ODE problem for a time interval = [0.0,1.0]
  ode_prob = ODEProblem(f_h,u0,(0.0,Tend))

  #Setup callback in order to fix CFL constant value
  cb = getCFLCallback(f_h, CFL)

  #Estimate an initial dt
  dt = update_dt!(u0, f_h, CFL)
  return ode_prob,mesh,cb, dt
end

mesh_ncells = [40,80,160,320]
t1 = get_conv_order_table(LaxWendroffScheme(),solve, get_problem, u_exact, mesh_ncells, Euler())

using PrettyTables 
header = ["M","e_tot","order"]
pretty_table(t1.data, header)