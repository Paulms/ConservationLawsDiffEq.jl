function solve{islinear,isstochastic,MeshType,F,F3,F4,F5}(
  prob::ConservationLawsProblem{islinear,isstochastic,MeshType,F,F3,F4,F5},
  alg::AbstractFVAlgorithm;
  TimeAlgorithm::OrdinaryDiffEqAlgorithm = SSPRK22(),force_cfl = true, kwargs...)

  #Unroll some important constants
  @unpack tspan,f,u0 = prob
  if !has_jac(f)
    f(::Type{Val{:jac}},x) = x -> ForwardDiff.jacobian(f,x)
  end

  # Semidiscretization
  ode_fv = get_semidiscretization(alg, prob)
  ode_prob = ODEProblem(ode_fv, u0, tspan)
  # Check CFL condition
  dt = update_dt(u0, ode_fv)
  condition(t,u,integrator) = force_cfl
  affect!(integrator) = set_proposed_dt!(integrator, update_dt(integrator.u, ode_fv))
  cb = DiscreteCallback(condition,affect!)
  sol = solve(ode_prob,TimeAlgorithm,dt=dt,callback=cb)

  return(FVSolution(sol.u,sol.t,prob,sol.retcode,sol.interp;dense = sol.dense))
end

function get_semidiscretization(alg::AbstractFVAlgorithm, prob::ConservationLawsProblem)
    @unpack f,CFL,numvars,mesh,u0 = prob
    fluxes = zeros(eltype(u0),numedges(mesh),numvars)
    dt = 0.0
    FVIntegrator(alg,mesh,f,CFL,numvars, fluxes, dt)
end

# function solve{islinear,isstochastic,MeshType,F,F3,F4,F5,F6}(
#   prob::ConservationLawsWithDiffusionProblem{islinear,isstochastic,MeshType,F,F3,F4,F5,F6},
#   alg::AbstractFVAlgorithm;
#   TimeAlgorithm::OrdinaryDiffEqAlgorithm = SSPRK22(),
#   save_everystep = false, kwargs...)
#
#   #Unroll some important constants
#   @unpack u0,f,CFL,tspan,numvars,mesh,DiffMat = prob
#   tend = tspan[end]
#   if !has_jac(f)
#     f(::Type{Val{:jac}},x) = x -> ForwardDiff.jacobian(f,x)
#   end
#
#   #Equation Loop
#   timeseries,ts,retcode,interp,dense=FV_solve(FVDiffIntegrator{typeof(alg),typeof(prob.mesh),typeof(tend),typeof(u0),
#   typeof(TimeAlgorithm),typeof(f),typeof(DiffMat)}(alg,prob.mesh,u0,f,DiffMat,CFL,
#   numvars,TimeAlgorithm,tend);save_everystep = save_everystep,
#   progress_steps=1000, kwargs...)
#
#   return(FVSolution(timeseries,ts,prob,retcode,interp;dense=dense))
# end
