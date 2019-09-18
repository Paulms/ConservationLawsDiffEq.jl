using ConservationLawsDiffEq
using OrdinaryDiffEq
using LinearAlgebra

const CFL = 0.5
const Tend = 1.0

Jf(u) = u
f(u) = u^2/2
f0(x) = sin(2*π*x)

function get_problem(N)
  mesh = Uniform1DFVMesh(N, 0.0, 1.0)
  #mesh = line_mesh(N, 0.0, 1.0)
  ConservationLawsProblem(f,f0,mesh,[Periodic()]; tspan = (0.0,Tend), Df = Jf)
end
#Run
prob = get_problem(200)
sol2 = solve(prob, LaxFriedrichsAlgorithm();TimeIntegrator = SSPRK22(), CFL = CFL, progress=true, save_everystep = false)
