function cdt(u::Matrix, CFL, dx,f)
  maxρ = 0
  N = size(u,1)
  for i in 1:N
    maxρ = max(maxρ, fluxρ(u[i,:],f))
  end
  CFL/(1/dx*maxρ)
end

function cdt(u::AbstractArray, CFL, dx, f, BB)
  maxρ = 0
  maxρB = 0
  N = size(u,1)
  for i in 1:N
    maxρ = max(maxρ, fluxρ(u[i,:],f))
    maxρB = max(maxρB, maximum(abs,eigvals(BB(u[i,:]))))
  end
  CFL/(1/dx*maxρ+1/(2*dx^2)*maxρB)
end

@inline function fluxρ(uj::Vector,f)
  maximum(abs,eigvals(f(Val{:jac}, uj)))
end

@inline function maxfluxρ(u::AbstractArray,f)
    maxρ = 0
    N = size(u,1)
    for i in 1:N
      maxρ = max(maxρ, fluxρ(u[i,:],f))
    end
    maxρ
end

function minmod(a,b,c)
  if (a > 0 && b > 0 && c > 0)
    min(a,b,c)
  elseif (a < 0 && b < 0 && c < 0)
    max(a,b,c)
  else
    zero(a)
  end
end

function minmod(a,b)
  0.5*(sign(a)+sign(b))*min(abs(a),abs(b))
end

#Common macros for all schemes
@def fv_uniform1Dmeshpreamble begin
  @unpack N,x,dx,bdtype = integrator.mesh
end
@def fv_diffdeterministicpreamble begin
  @unpack u0,Flux,DiffMat,CFL,M,TimeAlgorithm,tend = integrator
end

@def fv_deterministicpreamble begin
  @unpack u0,Flux,CFL,M,TimeAlgorithm,tend = integrator
end

@def fv_generalpreamble begin
  dt = zero(tend)
end

@def fv_postamble begin
  if timeIntegrator.sol.t[end] != tend
    savevalues!(timeIntegrator)
  end
  return(timeIntegrator.sol.u,timeIntegrator.sol.t, timeIntegrator.sol.retcode,timeIntegrator.sol.interp,timeIntegrator.sol.dense)
end

@def fv_timeloop begin
  #First dt
  dt = cdt(u0, CFL, dx, Flux)
  @fv_setup_time_integrator
  @inbounds for i in timeIntegrator
    dt = cdt(timeIntegrator.u, CFL, dx, Flux)
    set_proposed_dt!(timeIntegrator, dt)
  end
  @fv_postamble
end

@def fv_difftimeloop begin
  #First dt
  dt = cdt(u0, CFL, dx, Flux, DiffMat)
  @fv_setup_time_integrator
  @inbounds for i in timeIntegrator
    dt = cdt(timeIntegrator.u, CFL, dx, Flux, DiffMat)
    set_proposed_dt!(timeIntegrator, dt)
  end
  @fv_postamble
end

@def boundary_header begin
  uu = copy(uold)
  if bdtype == :PERIODIC
    uu = PeriodicMatrix(uu)
  elseif bdtype == :ZERO_FLUX
    uu = ZeroFluxMatrix(uu)
  end
end

@def boundary_update begin
  if bdtype == :ZERO_FLUX
    hh[1,:]=0.0; pp[1,:]=0.0
    hh[end,:]=0.0; pp[end,:]=0.0
  end
end

@def update_rhs begin
  for j = 1:N
    rhs[j,:] = - 1/dx * (hh[j+1,:]-hh[j,:]-(pp[j+1,:]-pp[j,:]))
  end
end

@def no_diffusion_term begin
  pp = zeros(N+1,M)
end

@def fv_setup_time_integrator begin
  rhs = zeros(u0)
  function semidiscretef(t,u,du)
    rhs!(du,u,N,M,dx,dt,bdtype)
  end
  prob = ODEProblem(semidiscretef, u0, (0.0,tend))
  timeIntegrator = init(prob, TimeAlgorithm;dt=dt, kwargs...)
end

# nflux must be capable of receiving vectors
@def fv_method_with_nflux_common begin
  @inline function rhs!(rhs, uold, N, M, dx, dt, bdtype)
    #Set ghost Cells
    @boundary_header
    # Numerical Fluxes
    hh = zeros(N+1,M)
    for j = 1:N+1
      hh[j,:] = nflux(uu[j-1,:], uu[j,:], dx, dt)
    end
    # Diffusion
    @no_diffusion_term
    @boundary_update
    @update_rhs
  end
  @fv_timeloop
end

#Low level schemes (Those who use custom time integration)
@def fv_nt_generalpreamble begin
  timeseries = Vector{typeof(u0)}(0)
  push!(timeseries,copy(u0))
  t = zero(tend)
  ts = Float64[t]
  saveat_vec = Vector{typeof(tend)}(0)
  if typeof(saveat) <: Number
    saveat_vec = convert(Vector{typeof(tend)},saveat:saveat:tend)
    # Exclude the endpoint because of floating point issues
  else
    saveat_vec =  convert(Vector{typeof(tend)},collect(saveat))
  end

  if !isempty(saveat_vec) && saveat_vec[end] < tend
    push!(saveat_vec, tend)
  end
  saveiter = 1
  savevec = true
  progress && (prog = Juno.ProgressBar(name=progressbar_name))
  percentage = 0
  limit = tend/10.0
  dt = zero(tend)
  u = copy(u0)
end

@def fv_nt_postamble begin
  progress && Juno.done(prog)
  if ts[end] != t
     push!(timeseries,copy(u))
     push!(ts,t)
  end
  timeseries,ts,:Default,LinearInterpolation(ts,timeseries),true
end

@def fv_nt_footer begin
  #TODO: interpolation to save at exact time?
  if (save_everystep && (i%timeseries_steps == 0)) || (savevec && t>saveat_vec[saveiter])
     saveiter = min(sum(saveat_vec .< t)+1, size(saveat_vec,1))
     if saveiter == size(saveat_vec,1)
       savevec = false
     end
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
