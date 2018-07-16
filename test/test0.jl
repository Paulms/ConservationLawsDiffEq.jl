# 1D Burgers Equation
# u_t+(0.5*u²)_{x}=0

using ConservationLawsDiffEq
using LinearAlgebra
include("burgers.jl")

const CFL = 0.5
const Tend = 1.0
const ul = 1.0
const ur = 0.0
const x0 = 0.0
const xl = -3.0
const xr = 3.0

prob1 = RiemannProblem(Burgers(), ul, ur, x0, 0.0)
sol_ana  = get_solution(prob1)

f(::Type{Val{:jac}},u::AbstractArray{T,1}) where {T} = Matrix(Diagonal(u))
f(u::AbstractArray) = u.^2/2

f0(x) = (x < x0) ? ul : ur

function get_problem(N)
  mesh = Uniform1DFVMesh(N,xl,xr,:DIRICHLET, :DIRICHLET)
  ConservationLawsProblem(f0,f,CFL,Tend,mesh)
end

prob = get_problem(50)
@time sol = solve(prob, FVSKTAlgorithm(); use_threads = false, save_everystep = true)
@test get_L1_error(sol_ana, sol) < 0.048
@test sol.t == [0.0,0.05999999999999994,0.11999999999999988,0.17999999999999983,0.23999999999999977,
                0.2999999999999997,0.35999999999999965,0.4199999999999996,0.47999999999999954,
                0.5399999999999995,0.5999999999999994,0.6599999999999994,0.7199999999999993,
                0.7799999999999992,0.8399999999999992,0.8999999999999991,0.9599999999999991,1.0]
@time sol1 = solve(prob, FVSKTAlgorithm(); use_threads = true, save_everystep = false)
@test get_L1_error(sol_ana, sol1) ≈ get_L1_error(sol_ana, sol)
println("Comparision with custom solve solution")
@time sol2 = fast_solve(prob, FVSKTAlgorithm();use_threads = false, save_everystep = true)
@test get_L1_error(sol_ana, sol2) ≈ get_L1_error(sol_ana, sol)
@test sol2.t == sol.t
@time sol3 = fast_solve(prob, FVSKTAlgorithm();use_threads = true, save_everystep = false)
@test get_L1_error(sol_ana, sol3) ≈ get_L1_error(sol_ana, sol)
println("Testing errors in the rest of schemes")
@time sol = solve(prob, LaxFriedrichsAlgorithm();use_threads = false, save_everystep = false)
@test get_L1_error(sol_ana, sol) < 0.24
@time sol1 = solve(prob, LaxFriedrichsAlgorithm();use_threads = true, save_everystep = false)
@test get_L1_error(sol_ana, sol1) ≈ get_L1_error(sol_ana, sol)
@time sol = solve(prob, LocalLaxFriedrichsAlgorithm();use_threads = false, save_everystep = false)
@test get_L1_error(sol_ana, sol) < 0.17
println("No threaded version of LLF")
@time sol = solve(prob, GlobalLaxFriedrichsAlgorithm();use_threads = false, save_everystep = false)
@test get_L1_error(sol_ana, sol) < 0.24
@time sol1 = solve(prob, GlobalLaxFriedrichsAlgorithm();use_threads = true, save_everystep = false)
@test get_L1_error(sol_ana, sol1) ≈ get_L1_error(sol_ana, sol)
#@time sol5 = solve(prob, LaxWendroff2sAlgorithm();progress=true, save_everystep = false)
@time sol = solve(prob, FVCompWENOAlgorithm();use_threads = false, TimeIntegrator = SSPRK33(), save_everystep = false)
@test get_L1_error(sol_ana, sol) < 0.041
@time sol1 = solve(prob, FVCompWENOAlgorithm();use_threads = true, TimeIntegrator = SSPRK33(), save_everystep = false)
@test get_L1_error(sol_ana, sol1) ≈ get_L1_error(sol_ana, sol)
@time sol = solve(prob, FVCompMWENOAlgorithm();use_threads = false, TimeIntegrator = SSPRK33(), save_everystep = false)
@test get_L1_error(sol_ana, sol) < 0.037
@time sol1 = solve(prob, FVCompMWENOAlgorithm();use_threads = true, TimeIntegrator = SSPRK33(), save_everystep = false)
@test get_L1_error(sol_ana, sol1) ≈ get_L1_error(sol_ana, sol)
@time sol = solve(prob, FVSpecMWENOAlgorithm();use_threads = false, save_everystep = false)
@test get_L1_error(sol_ana, sol) < 0.028
println("No threaded version of FVSpecMWENOAlgorithm")
@time sol = solve(prob, FVCUAlgorithm(); use_threads = false, save_everystep = false)
@test get_L1_error(sol_ana, sol) < 0.039
@time sol1 = solve(prob, FVCUAlgorithm(); use_threads = true, save_everystep = false)
@test get_L1_error(sol_ana, sol1) ≈ get_L1_error(sol_ana, sol)
@time sol = solve(prob, FVDRCUAlgorithm(); use_threads = false, save_everystep = false)
@test get_L1_error(sol_ana, sol) < 0.039
@time sol1 = solve(prob, FVDRCUAlgorithm(); use_threads = true, save_everystep = false)
@test get_L1_error(sol_ana, sol1) ≈ get_L1_error(sol_ana, sol)
@time sol = solve(prob, FVDRCU5Algorithm(); use_threads = false, save_everystep = true)
@test get_L1_error(sol_ana, sol) < 0.053
@time sol1 = solve(prob, FVDRCU5Algorithm(); use_threads = true, save_everystep = false)
@test get_L1_error(sol_ana, sol1) ≈ get_L1_error(sol_ana, sol)
@time sol = solve(prob, LaxWendroffAlgorithm(); use_threads = false, save_everystep = false)
@test get_L1_error(sol_ana, sol) < 0.043
@time sol1 = solve(prob, LaxWendroffAlgorithm(); use_threads = true, save_everystep = false)
@test get_L1_error(sol_ana, sol1) ≈ get_L1_error(sol_ana, sol)
basis=legendre_basis(3)
limiter! = DGLimiter(prob, basis, Linear_MUSCL_Limiter())
@time sol = solve(prob, DiscontinuousGalerkinScheme(basis, glf_num_flux); TimeIntegrator = SSPRK22(limiter!))
@test get_L1_error(sol_ana, sol) < 1.28
println("No threaded version of DG Scheme")
