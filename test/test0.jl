# Test 0: Solve 1D Burgers Equation
# u(x,t)_t+(0.5*u²(x,t))_{x}=0
# u(0,x) = f0(x)

@testset "1D Scalar Algorithms" begin

using ConservationLawsDiffEq
using LinearAlgebra
using OrdinaryDiffEq

include("burgers.jl")
CL = ConservationLawsDiffEq

CFL = 0.5
Tend = 1.0
ul = 1.0
ur = 0.0
x0 = 0.0

prob1 = RiemannProblem(Burgers(), ul, ur, x0, 0.0)
sol_ana  = get_solution(prob1)

# First define the problem data (Jacobian is optional but useful)
Jf(u) = u           #Jacobian
f(u) = u^2/2        #Flux function

f0(x) = (x < x0) ? ul : ur

# Now discretizate the domain
mesh = Uniform1DFVMesh(50, [-3.0, 3.0])

function get_problem(alg::CL.AbstractFVAlgorithm, use_threads, mesh, Tend,CFL)
  # Now get a explicit semidiscretization (discrete in space) du_h(t)/dt = f_h(u_h(t))
  f_h = getSemiDiscretization(f,alg,mesh,[Dirichlet(;side=:Both)]; Df = Jf, use_threads = use_threads,numvars = 1)

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
  prob,cb,dt = get_problem(alg, false, mesh, Tend,CFL)
  @time sol = solve(prob,SSPRK22(); dt = dt, callback = cb, save_everystep = false)
  @test get_L1_error(sol_ana, fv_solution(sol,mesh)) < maxerror
  if threaded_mode
    prob,cb,dt = get_problem(alg, true, mesh, Tend,CFL)
    @time sol1 = solve(prob,SSPRK22(); dt = dt, callback = cb, save_everystep = false)
    @test get_L1_error(sol_ana, fv_solution(sol1,mesh)) ≈ get_L1_error(sol_ana, fv_solution(sol,mesh))
  end
end



# Test Schemes relative error
test_Scheme(FVSKTScheme(), 0.048)
test_Scheme(LaxFriedrichsScheme(), 0.24)
test_Scheme(LocalLaxFriedrichsScheme(), 0.17, false)
test_Scheme(GlobalLaxFriedrichsScheme(), 0.24)
test_Scheme(LaxWendroffScheme(), 0.043)
test_Scheme(LaxWendroff2sScheme(), 0.088)
test_Scheme(FVCompWENOScheme(), 0.045)
test_Scheme(FVCompMWENOScheme(), 0.041)
test_Scheme(FVSpecMWENOScheme(), 0.028, false)
test_Scheme(FVCUScheme(), 0.039)
test_Scheme(FVDRCUScheme(), 0.039)
test_Scheme(FVDRCU5Scheme(), 0.053)

# Test dt and CFL callback
prob,cb,dt = get_problem(FVSKTScheme(), false, mesh, Tend,CFL)
@time sol = solve(prob,SSPRK22(); dt = dt, callback = cb, save_everystep = true)
@test sol.t == [0.0,0.05999999999999994,0.11999999999999988,0.17999999999999983,0.23999999999999977,
                0.2999999999999997,0.35999999999999965,0.4199999999999996,0.47999999999999954,
                0.5399999999999995,0.5999999999999994,0.6599999999999994,0.7199999999999993,
                0.7799999999999992,0.8399999999999992,0.8999999999999991,0.9599999999999991,1.0]
end
