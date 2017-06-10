function solve{islinear,isstochastic,MeshType,F,F3,F4,F5}(
  prob::ConservationLawsProblem{islinear,isstochastic,MeshType,F,F3,F4,F5},
  alg::AbstractFVAlgorithm;
  TimeAlgorithm::OrdinaryDiffEqAlgorithm = SSPRK22(),
  save_everystep = false,
  kwargs...)

  #Unroll some important constants
  @unpack u0,f,CFL,tspan,numvars,mesh = prob
  tend = tspan[end]
  if !has_jac(f)
    f(::Type{Val{:jac}},x) = x -> ForwardDiff.jacobian(f,x)
  end

  #Equation Loop
  timeseries,ts,retcode,interp,dense=FV_solve(FVIntegrator{typeof(alg),typeof(prob.mesh),typeof(tend),typeof(u0),
  typeof(TimeAlgorithm),typeof(f)}(alg,prob.mesh,u0,f,CFL,
  numvars,TimeAlgorithm,tend);save_everystep = save_everystep, progress_steps=1000, kwargs...)

  return(FVSolution(timeseries,ts,prob,retcode,interp;dense=dense))
end

function solve{islinear,isstochastic,MeshType,F,F3,F4,F5,F6}(
  prob::ConservationLawsWithDiffusionProblem{islinear,isstochastic,MeshType,F,F3,F4,F5,F6},
  alg::AbstractFVAlgorithm;
  TimeAlgorithm::OrdinaryDiffEqAlgorithm = SSPRK22(),
  save_everystep = false, kwargs...)

  #Unroll some important constants
  @unpack u0,f,CFL,tspan,numvars,mesh,DiffMat = prob
  tend = tspan[end]
  if !has_jac(f)
    f(::Type{Val{:jac}},x) = x -> ForwardDiff.jacobian(f,x)
  end

  #Equation Loop
  timeseries,ts,retcode,interp,dense=FV_solve(FVDiffIntegrator{typeof(alg),typeof(prob.mesh),typeof(tend),typeof(u0),
  typeof(TimeAlgorithm),typeof(f),typeof(DiffMat)}(alg,prob.mesh,u0,f,DiffMat,CFL,
  numvars,TimeAlgorithm,tend);save_everystep = save_everystep,
  progress_steps=1000, kwargs...)

  return(FVSolution(timeseries,ts,prob,retcode,interp;dense=dense))
end
