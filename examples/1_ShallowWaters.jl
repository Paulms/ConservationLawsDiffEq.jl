# 1D Shallow water equations with flat bottom
using ConservationLawsDiffEq
using OrdinaryDiffEq

const CFL = 0.1
const Tend = 0.2
const gr = 9.8

function Jf(u::AbstractVector)
  h = u[1]
  q = u[2]
  F =[0.0 1.0;-q^2/h^2+gr*h 2*q/h]
  F
end
f(u::AbstractVector) = [u[2];u[2]^2/u[1]+0.5*gr*u[1]^2]
f0(x) = x < 0.0 ? [2.0,0.0] : [1.0,0.0]

# function Nflux(ϕl::AbstractVector, ϕr::AbstractVector)
#   hl = ϕl[1]; hr = ϕr[1]
#   ul = ϕl[2]/ϕl[1]; ur = ϕr[2]/ϕr[1];
#   hm = 0.5*(hl+hr)
#   um = 0.5*(ul+ur)
#   return([hm*um;hm*um^2+0.5*gr*(0.5*(hl^2+hr^2))])
# end
#ve(u::AbstractVector) = [gr*u[1]-0.5*(u[2]/u[1])^2;u[2]/u[1]]


# Now discretizate the domain
mesh = Uniform1DFVMesh(100, [-5.0, 5.0])


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

#ode_prob, cb, dt = get_problem(FVTecnoScheme(Nflux;ve = ve, order=3), mesh)
#sol2 = solve(ode_prob,SSPRK22(); dt = dt, callback = cb)

#Plot
using Plots
u_h = fv_solution(sol, mesh; vars = 2)
plot(u_h, tidx=1, vars=1, lab="ho",line=(:dot,2))
plot!(u_h, vars=1,lab="KT h")
#u2_h = fv_solution(sol2, mesh)
#plot!(u2_h, vars=1,lab="Tecno h")
