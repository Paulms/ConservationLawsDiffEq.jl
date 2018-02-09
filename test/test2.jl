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
exact_sol(x::Float64, t::Float64) = hcat(0.5*(sin.(4*π*(-t+x))+sin.(4*π*(t+x))),
0.5*(sin.(4*π*(-t+x))-sin.(4*π*(t+x))))

function get_problem(N)
  mesh = Uniform1DFVMesh(N,-1.0,1.0,:PERIODIC, :PERIODIC)
  u0 = u0_func(cell_centers(mesh))
  ConservationLawsProblem(u0,f,CFL,Tend,mesh)
end
#Compile
prob = get_problem(20)

@time sol = fast_solve(prob, FVSKTAlgorithm();progress=true, use_threads=false)
@test get_L1_errors(exact_sol, sol) < 1.3
@time sol1 = fast_solve(prob, FVSKTAlgorithm();progress=true, use_threads=true)
@test get_L1_errors(exact_sol, sol1) ≈ get_L1_errors(exact_sol, sol)
@time sol = solve(prob, FVSKTAlgorithm();progress=true, use_threads=false)
@test get_L1_errors(exact_sol, sol) < 1.3
@time sol1 = solve(prob, FVSKTAlgorithm();progress=true, use_threads=true)
@test get_L1_errors(exact_sol, sol1) ≈ get_L1_errors(exact_sol, sol)
@time sol = solve(prob, FVTecnoAlgorithm(Nflux;order=3);progress=true, use_threads=false)
@test get_L1_errors(exact_sol, sol) < 1.3
println("No threaded version of TECNO scheme")
@time sol = solve(prob, FVCompWENOAlgorithm();progress=true, use_threads=false)
@test get_L1_errors(exact_sol, sol) < 1.01
@time sol1 = solve(prob, FVCompWENOAlgorithm();progress=true, use_threads=true)
@test get_L1_errors(exact_sol, sol1) ≈ get_L1_errors(exact_sol, sol)
@time sol = solve(prob, FVCompMWENOAlgorithm();progress=true, use_threads=false)
@test get_L1_errors(exact_sol, sol) < 0.72
@time sol1 = solve(prob, FVCompMWENOAlgorithm();progress=true, use_threads=true)
@test get_L1_errors(exact_sol, sol1) ≈ get_L1_errors(exact_sol, sol)
@time sol = solve(prob, FVSpecMWENOAlgorithm();progress=true, use_threads=false)
@test get_L1_errors(exact_sol, sol) < 0.72
println("No threaded version of FVSpecMWENOAlgorithm")
@time sol = solve(prob, FVCUAlgorithm(); use_threads = false, save_everystep = false)
@test get_L1_errors(exact_sol, sol) < 1.3
@time sol1 = solve(prob, FVCUAlgorithm(); use_threads = true, save_everystep = false)
@test get_L1_errors(exact_sol, sol1) ≈ get_L1_errors(exact_sol, sol)
@time sol = solve(prob, FVDRCUAlgorithm(); use_threads = false, save_everystep = false)
@test get_L1_errors(exact_sol, sol) < 1.3
@time sol1 = solve(prob, FVDRCUAlgorithm(); use_threads = true, save_everystep = false)
@test get_L1_errors(exact_sol, sol1) ≈ get_L1_errors(exact_sol, sol)
@time sol = solve(prob, FVDRCU5Algorithm(); use_threads = false, save_everystep = false)
@test get_L1_errors(exact_sol, sol) < 0.51
@time sol1 = solve(prob, FVDRCU5Algorithm(); use_threads = true, save_everystep = false)
@test get_L1_errors(exact_sol, sol1) ≈ get_L1_errors(exact_sol, sol)
