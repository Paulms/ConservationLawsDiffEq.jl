# One dimensional wave equation
using ConservationLawsDiffEq
using OrdinaryDiffEq
const CL = ConservationLawsDiffEq

const CFL = 0.45
const Tend = 1.0
const cc = 1.0

Jf(u::AbstractVector) = [0.0 cc;cc 0.0]
f(u::AbstractVector) = [0.0 cc;cc 0.0]*u
f0(x) = [sin(4*π*x),0.0]

Nflux(ϕl::AbstractVector, ϕr::AbstractVector) = 0.5*(f(ϕl)+f(ϕr))
exact_sol(x, t::Float64) = [0.5*(sin(4*π*(-t+x))+sin(4*π*(t+x))),
0.5*(sin(4*π*(-t+x))-sin(4*π*(t+x)))]

# Now discretizate the domain
mesh = Uniform1DFVMesh(50, [-1.0, 1.0])

function get_problem(alg, mesh)
  #Compute discrete initial data
  u0 = getInitialState(mesh,f0,use_threads = true)

  # Now get a explicit semidiscretization (discrete in space) du_h(t)/dt = f_h(u_h(t))
  f_h = getSemiDiscretization(f,alg,mesh,[Periodic()]; Df = Jf, use_threads = false,numvars = 2)

  #Setup ODE problem for a time interval = [0.0,1.0]
  ode_prob = ODEProblem(f_h,u0,(0.0,Tend))

  #Setup callback in order to fix CFL constant value
  cb = getCFLCallback(f_h, CFL)

  #Estimate an initial dt
  dt = update_dt!(u0, f_h, CFL)
  return ode_prob, cb, dt
end

ode_prob, cb, dt = get_problem(FVSKTScheme(), mesh)
sol = solve(ode_prob,SSPRK22(); dt = dt, callback = cb)

ode_prob, cb, dt = get_problem(FVTecnoScheme(Nflux;order=3), mesh)
sol2 = solve(ode_prob,SSPRK22(); dt = dt, callback = cb)

ode_prob, cb, dt = get_problem(FVCompWENOScheme(), mesh)
sol3 = solve(ode_prob,SSPRK22(); dt = dt, callback = cb)

ode_prob, cb, dt = get_problem(FVCompMWENOScheme(), mesh)
sol4 = solve(ode_prob,SSPRK22(); dt = dt, callback = cb)

ode_prob, cb, dt = get_problem(FVSpecMWENOScheme(), mesh)
sol5 = solve(ode_prob,SSPRK22(); dt = dt, callback = cb)

#Estimate errors
u1_h = fv_solution(sol, mesh; vars = 2)
u2_h = fv_solution(sol2, mesh; vars = 2)
u3_h = fv_solution(sol3, mesh; vars = 2)
u4_h = fv_solution(sol4, mesh; vars = 2)
u5_h = fv_solution(sol5, mesh; vars = 2)
get_L1_error(exact_sol,u1_h)
get_L1_error(exact_sol,u2_h)
get_L1_error(exact_sol,u3_h)
get_L1_error(exact_sol,u4_h)
get_L1_error(exact_sol,u5_h)

#Plot
using Plots
plot(u1_h, vars=1, lab="KT y",line = (:dot,2))
plot!(u2_h, vars=1,lab="Tecno y",line=(:dot,3))
plot!(u3_h, vars=1,lab="Comp WENO y",line=(:dot,3))
plot!(u4_h, vars=1,lab="Comp MWENO y",line=(:dot,3))
plot!(u5_h, vars=1,lab="Spec MWENO y",line=(:dot,3))
plot!(CL.cell_centers(mesh), [exact_sol(x,Tend)[1] for x in CL.cell_centers(mesh)],lab="Ref y")
