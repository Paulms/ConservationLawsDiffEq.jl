using ConservationLawsDiffEq
using LinearAlgebra
u_exact(x,t) = sin(4*pi*(x-t))
# Define the flux and flux Jacobian
Jf(u::AbstractVector{T}) where {T} = Matrix{T}(I,size(u,1),size(u,1))
f(u::AbstractVector) = u

# Initial condition (using integral cell averages)
f0(x) = sin(4*pi*x)

#Setup problem for a given N (number of cells/control volumenes on a uniform mesh)
#and given final time (Tend) with periodic boundary conditions
function get_problem(N; Tend = 2.0, CFL = 0.5)
  mesh = Uniform1DFVMesh(N,0.0,1.0,:PERIODIC, :PERIODIC)
  ConservationLawsProblem(f0,f,CFL,Tend,mesh;jac = Jf)
end

mesh_ncells = [40,80,160,320]
t1 = get_conv_order_table(LaxFriedrichsAlgorithm(), get_problem, u_exact, mesh_ncells; TimeIntegrator = Euler(), use_threads = true)
