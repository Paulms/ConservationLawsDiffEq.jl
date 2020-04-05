# One dimensional wave equation
# Test systems of conservation laws

@testset "1D Sytems Algorithms" begin

using ConservationLawsDiffEq
using OrdinaryDiffEq
CL = ConservationLawsDiffEq

CFL = 0.45
Tend = 1.0
cc = 1.0

Jf(u::AbstractVector) = [0.0 cc;cc 0.0]
f(u::AbstractVector) = [0.0 cc;cc 0.0]*u
f0(x) = [sin(4*π*x), 0.0]

Nflux(ϕl, ϕr) = 0.5*(f(ϕl)+f(ϕr))
exact_sol(x::Float64, t::Float64) = hcat(0.5*(sin.(4*π*(-t+x))+sin.(4*π*(t+x))),
0.5*(sin.(4*π*(-t+x))-sin.(4*π*(t+x))))

mesh = Uniform1DFVMesh(20, [-1.0, 1.0])

function get_problem(alg::CL.AbstractFVAlgorithm, use_threads, mesh, Tend,CFL)
  # Now get a explicit semidiscretization (discrete in space) du_h(t)/dt = f_h(u_h(t))
  f_h = getSemiDiscretization(f,alg,mesh,[Periodic()]; Df = Jf, use_threads = use_threads,numvars = 2)

  #Compute discrete initial data
  u0 = getInitialState(mesh,f0,use_threads = true)

  #Setup ODE problem for a time interval = [0.0,1.0]
  ode_prob = ODEProblem(f_h,u0,(0.0,Tend))

  #Setup callback in order to fix CFL constant value
  cb = getCFLCallback(f_h, CFL)

  #Estimate an initial dt
  dt = update_dt!(u0, f_h, CFL)
  return ode_prob, cb, dt
end

function test_scheme(alg, maxerror, threaded_mode=true)
  println("Testing ",split("$alg","(")[1],": ")
  prob,cb,dt = get_problem(alg, false, mesh, Tend,CFL)
  @time sol = solve(prob,SSPRK22(); dt = dt, callback = cb, save_everystep = false)
  @test get_L1_error(exact_sol, fv_solution(sol,mesh)) < maxerror
  if threaded_mode
    println("in threaded mode: ")
    prob,cb,dt = get_problem(alg, true, mesh, Tend,CFL)
    @time sol1 = solve(prob,SSPRK22(); dt = dt, callback = cb, save_everystep = false)
    @test get_L1_error(exact_sol, fv_solution(sol1,mesh)) ≈ get_L1_error(exact_sol, fv_solution(sol,mesh))
  end
end

# Test Schemes error
test_scheme(FVSKTScheme(), 1.16)
test_scheme(FVTecnoScheme(Nflux;order=2), 1.152, false) 
test_scheme(FVTecnoScheme(Nflux;order=3), 1.14, false) 
test_scheme(FVTecnoScheme(Nflux;order=5), 0.92, false)
test_scheme(LaxFriedrichsScheme(), 1.16)
test_scheme(LocalLaxFriedrichsScheme(), 1.16, false)
test_scheme(GlobalLaxFriedrichsScheme(), 1.16)
test_scheme(LaxWendroffScheme(), 37.0)    #way off
test_scheme(LaxWendroff2sScheme(), 1.152)
test_scheme(FVCompWENOScheme(), 0.94)
test_scheme(FVCompMWENOScheme(), 0.68)
test_scheme(FVSpecMWENOScheme(), 0.68, false)
test_scheme(FVCUScheme(), 1.16)
test_scheme(FVDRCUScheme(), 1.17)
test_scheme(FVDRCU5Scheme(), 0.48)

end
