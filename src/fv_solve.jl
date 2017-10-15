function solve{islinear,isstochastic,MeshType,F,F3,F4,F5}(
  prob::ConservationLawsProblem{islinear,isstochastic,MeshType,F,F3,F4,F5},
  alg::AbstractFVAlgorithm;
  TimeAlgorithm::OrdinaryDiffEqAlgorithm = SSPRK22(),use_threads = false, kwargs...)

  #Unroll some important constants
  @unpack tspan,f,u0 = prob
  if !has_jac(f)
    f(::Type{Val{:jac}},x) = x -> ForwardDiff.jacobian(f,x)
  end

  # Semidiscretization
  ode_fv = get_semidiscretization(alg, prob; use_threads = use_threads)
  ode_prob = ODEProblem(ode_fv, u0, tspan)
  # Check CFL condition
  ode_fv.dt = update_dt(u0, ode_fv)
  timeIntegrator = init(ode_prob, TimeAlgorithm;dt=ode_fv.dt,kwargs...)
  @inbounds for i in timeIntegrator
    ode_fv.dt = update_dt(timeIntegrator.u, ode_fv)
    set_proposed_dt!(timeIntegrator, ode_fv.dt)
  end
  return(FVSolution(timeIntegrator.sol.u,timeIntegrator.sol.t,prob,
  timeIntegrator.sol.retcode,timeIntegrator.sol.interp;dense = timeIntegrator.sol.dense))
end

function get_semidiscretization(alg::AbstractFVAlgorithm, prob::ConservationLawsProblem;use_threads=false)
    @unpack f,CFL,numvars,mesh,u0 = prob
    fluxes = zeros(eltype(u0),numedges(mesh),numvars)
    dt = 0.0
    FVIntegrator(alg,mesh,f,CFL,numvars, fluxes, dt, use_threads)
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
