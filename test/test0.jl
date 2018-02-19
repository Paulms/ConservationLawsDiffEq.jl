# 1D Burgers Equation
# u_t+(0.5*u²)_{x}=0

using ConservationLawsDiffEq
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

f(::Type{Val{:jac}},u::Vector) = diagm(u)
f(u::Vector) = u.^2/2

f0(x) = (x < x0) ? ul : ur

function get_problem(N)
  mesh = Uniform1DFVMesh(N,xl,xr,:DIRICHLET, :DIRICHLET)
  ConservationLawsProblem(f0,f,CFL,Tend,mesh)
end

prob = get_problem(50)
@time sol = solve(prob, FVSKTAlgorithm(); use_threads = false, save_everystep = false)
@test get_L1_error(sol_ana, sol) < 0.062
@time sol1 = solve(prob, FVSKTAlgorithm(); use_threads = true, save_everystep = false)
@test get_L1_error(sol_ana, sol1) ≈ get_L1_error(sol_ana, sol)
@time sol = fast_solve(prob, FVSKTAlgorithm();use_threads = false, save_everystep = false)
@test get_L1_error(sol_ana, sol) < 0.066
@time sol1 = fast_solve(prob, FVSKTAlgorithm();use_threads = true, save_everystep = false)
@test get_L1_error(sol_ana, sol1) ≈ get_L1_error(sol_ana, sol)
@time sol = solve(prob, LaxFriedrichsAlgorithm();use_threads = false, save_everystep = false)
@test get_L1_error(sol_ana, sol) < 0.26
@time sol1 = solve(prob, LaxFriedrichsAlgorithm();use_threads = true, save_everystep = false)
@test get_L1_error(sol_ana, sol1) ≈ get_L1_error(sol_ana, sol)
@time sol = solve(prob, LocalLaxFriedrichsAlgorithm();use_threads = false, save_everystep = false)
@test get_L1_error(sol_ana, sol) < 0.18
println("No threaded version of LLF")
@time sol = solve(prob, GlobalLaxFriedrichsAlgorithm();use_threads = false, save_everystep = false)
@test get_L1_error(sol_ana, sol) < 0.26
@time sol1 = solve(prob, GlobalLaxFriedrichsAlgorithm();use_threads = true, save_everystep = false)
@test get_L1_error(sol_ana, sol1) ≈ get_L1_error(sol_ana, sol)
#@time sol5 = solve(prob, LaxWendroff2sAlgorithm();progress=true, save_everystep = false)
@time sol = solve(prob, FVCompWENOAlgorithm();use_threads = false, TimeIntegrator = SSPRK33(), save_everystep = false)
@test get_L1_error(sol_ana, sol) < 0.055
@time sol1 = solve(prob, FVCompWENOAlgorithm();use_threads = true, TimeIntegrator = SSPRK33(), save_everystep = false)
@test get_L1_error(sol_ana, sol1) ≈ get_L1_error(sol_ana, sol)
@time sol = solve(prob, FVCompMWENOAlgorithm();use_threads = false, TimeIntegrator = SSPRK33(), save_everystep = false)
@test get_L1_error(sol_ana, sol) < 0.051
@time sol1 = solve(prob, FVCompMWENOAlgorithm();use_threads = true, TimeIntegrator = SSPRK33(), save_everystep = false)
@test get_L1_error(sol_ana, sol1) ≈ get_L1_error(sol_ana, sol)
@time sol = solve(prob, FVSpecMWENOAlgorithm();use_threads = false, save_everystep = false)
@test get_L1_error(sol_ana, sol) < 0.043
println("No threaded version of FVSpecMWENOAlgorithm")
@time sol = solve(prob, FVCUAlgorithm(); use_threads = false, save_everystep = false)
@test get_L1_error(sol_ana, sol) < 0.053
@time sol1 = solve(prob, FVCUAlgorithm(); use_threads = true, save_everystep = false)
@test get_L1_error(sol_ana, sol1) ≈ get_L1_error(sol_ana, sol)
@time sol = solve(prob, FVDRCUAlgorithm(); use_threads = false, save_everystep = false)
@test get_L1_error(sol_ana, sol) < 0.053
@time sol1 = solve(prob, FVDRCUAlgorithm(); use_threads = true, save_everystep = false)
@test get_L1_error(sol_ana, sol1) ≈ get_L1_error(sol_ana, sol)
@time sol = solve(prob, FVDRCU5Algorithm(); use_threads = false, save_everystep = false)
@test get_L1_error(sol_ana, sol) < 0.065
@time sol1 = solve(prob, FVDRCU5Algorithm(); use_threads = true, save_everystep = false)
@test get_L1_error(sol_ana, sol1) ≈ get_L1_error(sol_ana, sol)
@time sol = solve(prob, LaxWendroffAlgorithm(); use_threads = false, save_everystep = false)
@test get_L1_error(sol_ana, sol) < 0.057
@time sol1 = solve(prob, LaxWendroffAlgorithm(); use_threads = true, save_everystep = false)
@test get_L1_error(sol_ana, sol1) ≈ get_L1_error(sol_ana, sol)
function llf_num_flux(ul, ur)
    αl = fluxρ(ul, f)
    αr = fluxρ(ur, f)
    αk = max(αl, αr)
    return 0.5*(f(ul)+f(ur))-αk*(ur-ul)
end
basis=legendre_basis(3)
limiter! = DGLimiter(prob.mesh, basis, Linear_MUSCL_Limiter())
@time sol = solve(prob, DiscontinuousGalerkinScheme(basis, llf_num_flux); TimeIntegrator = SSPRK22(limiter!))
@test get_L1_error(sol_ana, sol) < 1.4
println("No threaded version of DG Scheme")
