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

function fv_solution(sol::AbstractODESolution{T,N}, mesh) where {T,N}
    FVSolution{T,N,typeof(mesh)}(sol, mesh)
  end