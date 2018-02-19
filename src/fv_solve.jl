function solve(
  prob::AbstractConservationLawProblem,
  alg::AbstractFVAlgorithm;
  TimeIntegrator::OrdinaryDiffEqAlgorithm = SSPRK22(),
  average_initial_data = true, use_threads = false, kwargs...)

  #Unroll some important constants
  @unpack tspan,f,f0, mesh = prob
  #Compute initial data
  N = numcells(mesh)
  u0 = zeros(mesh.N, prob.numvars)
  compute_initial_data!(u0, f0, average_initial_data, mesh, Val{use_threads})

  if !has_jac(f)
    f(::Type{Val{:jac}},x) = x -> ForwardDiff.jacobian(f,x)
  end

  # Semidiscretization
  ode_fv = get_semidiscretization(alg, prob; use_threads = use_threads)
  ode_prob = ODEProblem(ode_fv, u0, tspan)
  # Check CFL condition
  ode_fv.dt = update_dt(u0, ode_fv)
  timeIntegrator = init(ode_prob, TimeIntegrator;dt=ode_fv.dt,kwargs...)
  @inbounds for i in timeIntegrator
    ode_fv.dt = update_dt(timeIntegrator.u, ode_fv)
    set_proposed_dt!(timeIntegrator, ode_fv.dt)
  end
  return(FVSolution(timeIntegrator.sol.u,timeIntegrator.sol.t,prob,
  timeIntegrator.sol.retcode,timeIntegrator.sol.interp;dense = timeIntegrator.sol.dense))
end

function initial_data_inner_loop!(u0, f0, average_initial_data, faces, centers, mesh, i)
    if average_initial_data
        u0[i,:] = num_integrate(f0,faces[i], faces[i+1])/volume(i, mesh)
    else
        u0[i,:] = f0(centers[i])
    end
end
function compute_initial_data!(u0, f0, average_initial_data, mesh, ::Type{Val{true}})
    faces = cell_faces(mesh)
    centers = cell_centers(mesh)
    Threads.@threads for i in 1:numcells(mesh)
        initial_data_inner_loop!(u0, f0, average_initial_data, faces, centers, mesh, i)
    end
end
function compute_initial_data!(u0, f0, average_initial_data, mesh, ::Type{Val{false}})
    faces = cell_faces(mesh)
    centers = cell_centers(mesh)
    for i in 1:numcells(mesh)
        initial_data_inner_loop!(u0, f0, average_initial_data, faces, centers, mesh, i)
    end
end

function get_semidiscretization(alg::AbstractFVAlgorithm, prob::ConservationLawsProblem;use_threads=false)
    @unpack f0, f,CFL,numvars,mesh = prob
    fluxes = zeros(eltype(f0(cell_faces(mesh)[1])),numedges(mesh),numvars)
    dt = 0.0
    FVIntegrator(alg,mesh,f,CFL,numvars, fluxes, dt, use_threads)
end

function get_semidiscretization(alg::AbstractFVAlgorithm, prob::ConservationLawsWithDiffusionProblem;use_threads=false)
    @unpack f0,f,CFL,numvars,mesh, DiffMat = prob
    fluxes = zeros(eltype(f0(cell_faces(mesh)[1])),numedges(mesh),numvars)
    dt = 0.0
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

  N = mesh.N
  NC = prob.numvars
  NN = basis.order+1
  #Assign Initial values (u0 = φₕ⋅u0ₘ)
  u0ₘ = zeros(NN*NC, N)
  for i = 1:N
    for j = 1:NC
      value = project_function(f0,basis,(cell_faces(mesh)[i],cell_faces(mesh)[i+1]); component = j)
      u0ₘ[(j-1)*NN+1:NN*j,i] = value.param
    end
  end

  #build inverse of mass matrix
  M_inv = get_local_inv_mass_matrix(basis, mesh)

  #Time loop
  #First dt
  u0ₕ = reconstruct_u(u0ₘ, basis.φₕ, NC)
  dt = update_dt(alg, u0ₕ, f, prob.CFL, mesh)
  # Setup time integrator
  semidiscretef(du,u,p,t) = residual!(du, u, basis, mesh, f, riemann_solver, M_inv,NC)
  ode_prob = ODEProblem(semidiscretef, u0ₘ, prob.tspan)
  timeIntegrator = init(ode_prob, TimeIntegrator;dt=dt, kwargs...)
  @inbounds for i in timeIntegrator
    uₕ = reconstruct_u(timeIntegrator.u, basis.φₕ, NC)
    dt = update_dt(alg, uₕ, f, prob.CFL, mesh)
    set_proposed_dt!(timeIntegrator, dt)
  end
  if timeIntegrator.sol.t[end] != prob.tspan[end]
    savevalues!(timeIntegrator)
  end
  return build_solution(timeIntegrator.sol,basis,prob, NC)
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
  average_initial_data = true,
  save_everystep = false,
  iterations=1000000,
  TimeIntegrator=:SSPRK22,
  progress::Bool=false,progressbar_name="FV",
  use_threads = false, kwargs...)

  #Unroll some important constants
  @unpack tspan,f,f0, mesh = prob

  #Compute initial data
  N = numcells(mesh)
  u0 = zeros(mesh.N, prob.numvars)
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
