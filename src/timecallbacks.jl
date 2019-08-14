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
function update_dt!(u,fv::FVIntegrator, CFL)
  fv.dt = update_dt(fv.alg, u, fv.Flux, CFL, fv.mesh)
  fv.dt
end

function getCFLCallback(f_sd, CFL)
    dtFE(u,p,t) = update_dt!(u, f_sd, CFL)
    StepsizeLimiter(dtFE;safety_factor=1.0,max_step=true,cached_dtcache=0.0)
end
