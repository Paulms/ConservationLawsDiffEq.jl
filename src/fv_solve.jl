function solve(
  prob::AbstractConservationLawProblem,
  alg::AbstractFVAlgorithm;
  TimeIntegrator::OrdinaryDiffEqAlgorithm = SSPRK22(),
  average_initial_data::Bool = true, use_threads::Bool = false, kwargs...)

  #Unroll some important constants
  @unpack tspan,f,f0, mesh = prob
  #Compute initial data
  N = numcells(mesh)
  NType = eltype(f0(cell_faces(mesh)[1]))
  u0 = MMatrix{mesh.N, prob.numvars,NType}()
  compute_initial_data!(u0, f0, average_initial_data, mesh, Val{use_threads})

  if !has_jac(f)
    f(::Type{Val{:jac}},x) = x -> ForwardDiff.jacobian(f,x)
  end

  # Semidiscretization
  ode_fv = get_semidiscretization(alg, prob; use_threads = use_threads)
  ode_prob = ODEProblem(ode_fv, u0, tspan)
  # Set update_dt function
  dtFE(u,p,t) = update_dt!(u, ode_fv)
  cb = StepsizeLimiter(dtFE;safety_factor=one(NType),max_step=true,cached_dtcache=zero(eltype(tspan)))
  # Check initial CFL condition
  update_dt!(u0, ode_fv)
  # Call ODE solve method
  sol = solve(ode_prob,TimeIntegrator, dt = ode_fv.dt,callback=cb; kwargs...)
  return(FVSolution(sol.u,sol.t,prob,
  sol.retcode,sol.interp;dense = sol.dense))
end

function initial_data_inner_loop!(u0, f0, average_initial_data, mesh, i)
    if average_initial_data
        faces = cell_faces(mesh)
        u0[i,:] = num_integrate(f0,faces[i], faces[i+1])/cell_volume(i, mesh)
    else
        centers = cell_centers(mesh)
        u0[i,:] = f0(centers[i])
    end
end
function compute_initial_data!(u0, f0, average_initial_data, mesh, ::Type{Val{true}})
    Threads.@threads for i in 1:numcells(mesh)
        initial_data_inner_loop!(u0, f0, average_initial_data, mesh, i)
    end
end
function compute_initial_data!(u0, f0, average_initial_data, mesh, ::Type{Val{false}})
    for i in 1:numcells(mesh)
        initial_data_inner_loop!(u0, f0, average_initial_data, mesh, i)
    end
end

function get_semidiscretization(alg::AbstractFVAlgorithm, prob::ConservationLawsProblem;use_threads::Bool=false)
    @unpack f0, f,CFL,numvars,mesh,tspan = prob
    fluxes = MMatrix{numedges(mesh),numvars,eltype(f0(cell_faces(mesh)[1]))}()
    dt = zero(eltype(tspan))
    FVIntegrator(alg,mesh,f,CFL,numvars, fluxes, dt, use_threads)
end

function get_semidiscretization(alg::AbstractFVAlgorithm, prob::ConservationLawsWithDiffusionProblem;use_threads::Bool=false)
    @unpack f0,f,CFL,numvars,mesh,DiffMat,tspan = prob
    fluxes = MMatrix{numedges(mesh),numvars,eltype(f0(cell_faces(mesh)[1]))}()
    dt = zero(eltype(tspan))
    FVDiffIntegrator(alg,mesh,f,DiffMat,CFL,numvars, fluxes, dt, use_threads)
end
##############Solve method for DG scheme ##################
"Solve scalar 1D conservation laws problems with DG Scheme"
function solve(
  prob::AbstractConservationLawProblem,
  alg::AbstractFEAlgorithm;
  TimeIntegrator::OrdinaryDiffEqAlgorithm = SSPRK22(),use_threads = false, kwargs...)

  # Unpack some useful variables
  @unpack basis, riemann_solver = alg
  #Unroll some important constants
  @unpack tspan,f,f0, mesh = prob

  N = numcells(mesh)
  NC = prob.numvars
  NN = basis.order+1
  #Assign Initial values
  xg = zeros(NN,N)
  for i in 1:N
    b = cell_faces(mesh)[i+1]; a=cell_faces(mesh)[i]
    xg[:,i] = reference_to_interval(basis.nodes,(a,b))
  end
  NType = eltype(f0(cell_faces(mesh)[1]))
  u0 = zeros(NType,NN*NC, N)
  for i = 1:N
      for k = 1:NN
        u0[k:NN:NN*NC,i] = f0(xg[k,i])
      end
  end
  # Setup time integrator
  semidiscretef(du,u,p,t) = residual!(du, u, basis, mesh, alg, f, riemann_solver, NC, Val{use_threads})
  ode_prob = ODEProblem(semidiscretef, u0, prob.tspan)
  # Set update_dt function
  function dtFE(u,p,t)
      uₕ = flat_u(u, basis.order,NC)
      update_dt(alg, uₕ, f, prob.CFL, mesh)
  end
  cb = StepsizeLimiter(dtFE;safety_factor=one(NType),max_step=true,cached_dtcache=zero(eltype(tspan)))
  # Check initial CFL condition
  u0ₕ = flat_u(u0, basis.order, NC)
  dt = update_dt(alg, u0ₕ, f, prob.CFL, mesh)
  # Call ODE solve method
  sol = solve(ode_prob,TimeIntegrator, dt = dt,callback=cb; kwargs...)
  return build_solution(sol,xg,basis,prob, NC)
end

######### Legacy solve method (easy to debug)
@def fv_generalpreamble begin
  progress && (prog = Juno.ProgressBar(name=progressbar_name))
  percentage = 0
  limit = tend/10.0
end

@def fv_postamble begin
  progress && Juno.done(prog)
  if ts[end] != t
     push!(timeseries,copy(u))
     push!(ts,t)
  end
end

@def fv_footer begin
    if save_everystep
       push!(timeseries,copy(u))
       push!(ts,t)
    end
  if progress && t>limit
    percentage = percentage + 10
    limit = limit +tend/10.0
    Juno.msg(prog,"dt="*string(dt))
    Juno.progress(prog,percentage/100.0)
  end
  if (t>=tend)
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
  average_initial_data::Bool = true,
  save_everystep::Bool = false,
  iterations::Int=1000000,
  TimeIntegrator=:SSPRK22,
  progress::Bool=false,progressbar_name="FV",
  use_threads::Bool = false, kwargs...)

  #Unroll some important constants
  @unpack tspan,f,f0, mesh = prob

  #Compute initial data
  N = numcells(mesh)
  u0 = Matrix{eltype(f0(cell_faces(mesh)[1]))}(mesh.N, prob.numvars)
  compute_initial_data!(u0, f0, average_initial_data, mesh, Val{use_threads})

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
  update_dt!(u0, ode_fv)

  u = copy(u0)
  rhs = zeros(u0)
  @fv_generalpreamble
  @inbounds for i=1:iterations
    update_dt!(u, ode_fv)
    if t + ode_fv.dt > tend; ode_fv.dt = tend - t; end
    t += ode_fv.dt
    @fv_deterministicloop
    @fv_footer
  end
  @fv_postamble

  return(FVSolution(timeseries,ts,prob,:Default,LinearInterpolation(ts,timeseries);dense = false))

end
