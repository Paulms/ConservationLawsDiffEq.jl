function initial_data_inner_loop!(u0, f0, nodes, method, mesh, i)
    if method == :average
        u0[:,i] .= num_integrate(f0,nodes[i], nodes[i+1])/cell_volume(mesh,i)
    elseif method == :eval_in_centers
        u0[:,i] .= f0((nodes[i]+nodes[i+1])/2)
    else
        error("invalid initial data processing method: ", method)
    end
    nothing
end

function getInitialState(mesh, f0; method =:average, use_threads = false, MType = Float64)
    N = getncells(mesh)
    numvars = size(f0(getnodecoords(mesh, 1)[1]),1)
    u0 = MMatrix{numvars,N,MType}(undef)
    nodes = get_nodes_matrix(mesh)
    if use_threads
        Threads.@threads for i in 1:getncells(mesh)
            initial_data_inner_loop!(u0, f0, nodes, method, mesh, i)
        end
    else
        for i in 1:getncells(mesh)
            initial_data_inner_loop!(u0, f0, nodes, method, mesh, i)
        end
    end
    return numvars > 1 ? u0 : u0[1,:]
end

function getSemiDiscretization(f,alg::AbstractFVAlgorithm,
    mesh,dbcs; Df = nothing, numvars = 1,
    Type = Float64, use_threads = false, dt = 0.0)
    probtype = numvars > 1 ? GeneralProblem() : ScalarProblem()
    internal_mesh = mesh_setup(mesh, dbcs,probtype)
    fluxes = numvars > 1 ? MMatrix{numvars,getnnodes(mesh),Type}(undef) : MVector{getnnodes(mesh),Type}(undef)
    Flux = numvars > 1 ? flux_function(f, Df) : scalar_flux_function(f, Df)
    FVIntegrator(alg,internal_mesh,Flux,numvars, fluxes, dt, use_threads)
end

function solve(
  prob::AbstractConservationLawProblem,
  alg::AbstractFVAlgorithm;
  TimeIntegrator = nothing,
  initial_data_method = :average, dt = nothing, CFL = nothing, use_threads::Bool = false, kwargs...)

  #Unroll some important constants
  tspan = prob.tspan; f = prob.f; f0 = prob.f0; mesh = prob.mesh
  bcs = prob.bcs; Df = prob.Df; numvars = prob.numvars
  #Compute initial data
  u0 = getInitialState(mesh,f0,method=initial_data_method, use_threads = use_threads)

  #Get semidiscretization
  f_sd = getSemiDiscretization(f,alg,mesh,bcs; Df = Df, use_threads = use_threads,
  numvars = numvars, Type = eltype(u0))

  #get ode problem
  ode_prob = ODEProblem(f_sd,u0,tspan)

  # Setup dt update method
  setupCallback = false
  cb = nothing
  #Setup initial dt
  if dt == nothing && CFL == nothing
      #Assume an adaptative ODE solver is being used
      setupCallback = true
      cb = get_adaptative_callback(f_sd)
  elseif CFL != nothing
      #Assume dt is limited by CFL condition (ignore dt)
      if dt != nothing
          warn("dt value will be ignored since CFL condition is given")
      end
      cb = getCFLCallback(f_sd, CFL)
      dt = update_dt!(u0, f_sd, CFL)
      setupCallback = true
  end
  #In any other case left ODE solver handle dt parameters

  # Call ODE Solver
  if dt == nothing
      if setupCallback
          sol = solve(ode_prob,TimeIntegrator; callback = cb, kwargs...)
      else
          sol = solve(ode_prob,TimeIntegrator; kwargs...)
      end
  else
      if setupCallback
          sol = solve(ode_prob,TimeIntegrator; dt = dt, callback = cb, kwargs...)
      else
          sol = solve(ode_prob,TimeIntegrator; dt = dt, kwargs...)
      end
  end
  return _build_solution(sol, prob)
end

function _build_solution(sol::AbstractODESolution{T,N}, prob::AbstractConservationLawProblem{A1,MeshType}) where {T,N,A1,MeshType}
  FVSolution{T,N,typeof(sol.u),typeof(sol.t),typeof(prob),MeshType,typeof(sol.interp)}(sol.u,sol.t,prob,sol.dense,0,sol.interp,sol.retcode)
end
