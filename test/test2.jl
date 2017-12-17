# One dimensional wave equation
# Test systems of conservation laws
using ConservationLawsDiffEq

const CFL = 0.45
const Tend = 1.0
const cc = 1.0

f(::Type{Val{:jac}},u::Vector) = [0.0 cc;cc 0.0]
f(u::Vector) = [0.0 cc;cc 0.0]*u

function u0_func(xx)
  N = size(xx,1)
  uinit = zeros(N, 2)
  uinit[:,1] = sin.(4*π*xx)
  return uinit
end

Nflux(ϕl::Vector, ϕr::Vector) = 0.5*(f(ϕl)+f(ϕr))
exact_sol(x::Vector, t::Float64) = hcat(0.5*(sin.(4*π*(-t+x))+sin.(4*π*(t+x))),
0.5*(sin.(4*π*(-t+x))-sin.(4*π*(t+x))))

function get_problem(N)
  mesh = Uniform1DFVMesh(N,-1.0,1.0,:PERIODIC, :PERIODIC)
  u0 = u0_func(cell_centers(mesh))
  ConservationLawsProblem(u0,f,CFL,Tend,mesh)
end
#Compile
prob = get_problem(20)

@time sol = solve(prob, FVSKTAlgorithm();progress=true)
@test get_L1_errors(sol, exact_sol; nvar = 1) < 0.61
@time sol2 = solve(prob, FVTecnoAlgorithm(Nflux;order=3);progress=true)
@test get_L1_errors(sol2, exact_sol; nvar = 1) < 0.61
@time sol3 = solve(prob, FVCompWENOAlgorithm();progress=true)
@test get_L1_errors(sol3, exact_sol; nvar = 1) < 0.43
@time sol4 = solve(prob, FVCompMWENOAlgorithm();progress=true)
@test get_L1_errors(sol4, exact_sol; nvar = 1) < 0.32
@time sol5 = solve(prob, FVSpecMWENOAlgorithm();progress=true)
@test get_L1_errors(sol5, exact_sol; nvar = 1) < 0.32
@time sol6 = solve(prob, FVCUAlgorithm(); use_threads = true, save_everystep = false)
@test get_L1_errors(sol6, exact_sol; nvar = 1) < 0.61
@time sol7 = solve(prob, FVDRCUAlgorithm(); use_threads = true, save_everystep = false)
@test get_L1_errors(sol7, exact_sol; nvar = 1) < 0.6
@time sol8 = solve(prob, FVDRCU5Algorithm(); use_threads = true, save_everystep = false)
@test get_L1_errors(sol8, exact_sol; nvar = 1) < 0.6
