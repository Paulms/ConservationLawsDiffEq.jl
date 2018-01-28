function solve(
  prob::AbstractConservationLawProblem,
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

function get_semidiscretization(alg::AbstractFVAlgorithm, prob::ConservationLawsWithDiffusionProblem;use_threads=false)
    @unpack f,CFL,numvars,mesh,u0, DiffMat = prob
    fluxes = zeros(eltype(u0),numedges(mesh),numvars)
    dt = 0.0
    FVDiffIntegrator(alg,mesh,f,DiffMat,CFL,numvars, fluxes, dt, use_threads)
end

######### Legacy solve method (easy to debug)
@def fv_generalpreamble begin
  progress && (prog = Juno.ProgressBar(name=progressbar_name))
  percentage = 0
  limit = tend/10.0
  timeStep = tend/timeseries_steps
  timeLimit = timeStep
end

@def fv_postamble begin
  progress && Juno.done(prog)
  if ts[end] != t
     push!(timeseries,copy(u))
     push!(ts,t)
  end
end

@def fv_footer begin
  if save_everystep && t>timeLimit
     push!(timeseries,copy(u))
     push!(ts,t)
     timeLimit = timeLimit + timeStep
  end
  if progress && t>limit
    percentage = percentage + 10
    limit = limit +tend/10.0
    Juno.msg(prog,"dt="*string(dt))
    Juno.progress(prog,percentage/100.0)
  end
  if (t>tend)
    break
  end
end

# Custom time integrators
@def fv_deterministicloop begin
  uold = copy(u)
  dt = ode_fv.dt
  if (TimeIntegrator == :FORWARD_EULER)
    ode_fv(rhs,uold,nothing,t)
    u = uold + dt*rhs
  elseif (TimeIntegrator == :SSPRK22)
    #FIRST Step
    ode_fv(rhs,uold,nothing,t)
    u = 0.5*(uold + dt*rhs)
    #Second Step
    ode_fv(rhs,uold + dt*rhs,nothing, t)
    u = u + 0.5*(uold + dt*rhs)
  elseif (TimeIntegrator == :RK4)
    #FIRST Step
    ode_fv(rhs,uold,nothing,t)
    u = uold + dt/6*rhs
    #Second Step
    ode_fv(rhs,uold+dt/2*rhs,nothing,t)
    u = u + dt/3*rhs
    #Third Step
    ode_fv(rhs,uold+dt/2*rhs,nothing,t)
    u = u + dt/3*rhs
    #Fourth Step
    ode_fv(rhs,uold+dt*rhs,nothing,t)
    u = u + dt/6 *rhs
  elseif (TimeIntegrator == :SSPRK33)
    ode_fv(rhs,uold,nothing,t)
    tmp = uold + dt*rhs
    ode_fv(rhs,tmp,nothing,t)
    tmp = (3*uold + tmp + dt*rhs) / 4
    ode_fv.dt = ode_fv.dt/2
    ode_fv(rhs,tmp,nothing,t)
    ode_fv.dt = ode_fv.dt*2
    u = (uold + 2*tmp + 2*dt*rhs) / 3
  elseif (TimeIntegrator == :SSPRK104)
    dt_6 = dt/6
    dt_3 = dt/3
    dt_2 = dt/2
    ode_fv(rhs,uold,nothing,t)
    tmp = uold + dt_6 * rhs # u₁
    ode_fv.dt = dt_6
    ode_fv(rhs,tmp,nothing,t)
    tmp = tmp + dt_6 * rhs # u₂
    ode_fv.dt = dt_3
    ode_fv(rhs,tmp,nothing,t)
    tmp = tmp + dt_6 * rhs # u₃
    ode_fv.dt = dt_2
    ode_fv(rhs,tmp,nothing,t)
    u₄ = tmp + dt_6 * rhs # u₄
    k₄ = zeros(rhs)
    ode_fv.dt = dt_3
    ode_fv(k₄,u₄,nothing,t)
    tmp = (3*uold + 2*u₄ + 2*dt_6 * k₄) / 5 # u₅
    ode_fv(rhs,tmp,nothing,t)
    tmp = tmp + dt_6 * rhs # u₆
    ode_fv.dt = dt_2
    ode_fv(rhs,tmp,nothing,t)
    tmp = tmp + dt_6 * rhs # u₇
    ode_fv.dt = dt_3
    ode_fv(rhs,tmp,nothing,t)
    tmp = tmp + dt_6 * rhs # u₈
    ode_fv.dt = 5*dt_6
    ode_fv(rhs,tmp,nothing,t)
    tmp = tmp + dt_6 * rhs # u₉
    ode_fv.dt = dt
    ode_fv(rhs,tmp,nothing,t)
    u = (uold + 9*(u₄ + dt_6*k₄) + 15*(tmp + dt_6*rhs)) / 25
  else
    throw("Time integrator not defined...")
  end
end

function fast_solve(
  prob::AbstractConservationLawProblem,
  alg::AbstractFVAlgorithm;
  timeseries_steps::Int = 100,
  save_everystep::Bool = false,
  iterations=1000000,
  TimeIntegrator=:SSPRK22,
  progress::Bool=false,progressbar_name="FV",
  use_threads = false, kwargs...)

  #Unroll some important constants
  @unpack tspan,f,u0 = prob
  if !has_jac(f)
    f(::Type{Val{:jac}},x) = x -> ForwardDiff.jacobian(f,x)
  end

  # Semidiscretization
  ode_fv = get_semidiscretization(alg, prob; use_threads = use_threads)

  #Setup timeseries
  t = tspan[1]
  timeseries = Vector{typeof(u0)}(0)
  push!(timeseries,copy(u0))
  ts = Float64[t]
  tend = tspan[end]

  # Check CFL condition
  ode_fv.dt = update_dt(u0, ode_fv)

  u = copy(u0)
  rhs = zeros(u0)
  @fv_generalpreamble
  @inbounds for i=1:iterations
    ode_fv.dt = update_dt(u, ode_fv)
    t += ode_fv.dt
    @fv_deterministicloop
    @fv_footer
  end
  @fv_postamble

  return(FVSolution(timeseries,ts,prob,:Default,LinearInterpolation(ts,timeseries);dense = false))

end
