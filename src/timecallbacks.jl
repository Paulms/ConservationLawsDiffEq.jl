# Callback for adaptative ODE methods
struct AdaptFVCallbackAffect{F}
    f_sd::F
end
function (f::AdaptFVCallbackAffect)(integrator)
    f_sd.dt = get_proposed_dt(integrator)
end

function get_adaptative_callback(f_sd)
    condition = (u,t,integrator) -> true
    affect! = AdaptFVCallbackAffect(f_sd)
    return DiscreteCallback(condition,affect!,save_positions=(false, false))
end

# Callback to enforce CFL condition on dt
function update_dt(alg::AbstractFVAlgorithm,u,Flux,
    CFL,mesh)
  maxρ = zero(eltype(u))
  dx = cell_volume(mesh, 1)
  for i in cell_indices(mesh)
    maxρ = max(maxρ, fluxρ(value_at_cell(u,i,mesh), Flux))
  end
  CFL/(1/dx*maxρ)
end
function update_dt!(u,fv::FVIntegrator)
  fv.dt = update_dt(fv.alg, u, fv.Flux, fv.CFL, fv.mesh)
  fv.dt
end

function getCFLCallback(f_sd)
    dtFE(u,p,t) = update_dt!(u, f_sd)
    StepsizeLimiter(dtFE;safety_factor=1.0,max_step=true,cached_dtcache=0.0)
end
