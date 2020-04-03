abstract type AbstractFVAlgorithm end

function compute_fluxes!(fluxes, Flux, u, mesh, dt, alg::AbstractFVAlgorithm, noscalar::Bool, ::Type{Val{false}})
    # This works assuming uniform mesh TODO: trait for uniform meshes
    dx = cell_volume(mesh, 1)
    #update vector
    for j in node_indices(mesh)
        if noscalar
            fluxes[:,j] .= update_flux_value(u,j,dt,dx,mesh,Flux,alg)
        else
            fluxes[j] = update_flux_value(u,j,dt,dx,mesh,Flux,alg)
        end
    end
end

function compute_fluxes!(fluxes, Flux, u, mesh, dt, alg::AbstractFVAlgorithm, noscalar::Bool, ::Type{Val{true}})
    # This works assuming uniform mesh TODO: trait for uniform meshes
    dx = cell_volume(mesh, 1)
    #update vector
    Threads.@threads for j in node_indices(mesh)
        if noscalar
            fluxes[:,j] .= update_flux_value(u,j,dt,dx,mesh,Flux,alg)
        else
            fluxes[j] = update_flux_value(u,j,dt,dx,mesh,Flux,alg)
        end
    end
end
