struct fvflux{ftype,dftype}
    flux::ftype
    Dflux::dftype
end

(F::fvflux)(u) = F.flux(u)

function scalar_flux_function(f, Df)
    if Df == nothing
        Df = x -> ForwardDiff.derivative(f,x)
    end
    fvflux(f,Df)
end

function flux_function(f, Df)
    if Df == nothing
        Df = x -> ForwardDiff.jacobian(f,x)
    end
    fvflux(f,Df)
end

@inline evalJacobian(f::fvflux, uj) = f.Dflux(uj)

function fluxρ(uj,f::fvflux) where {T}
        maximum(abs,eigvals(f.Dflux(uj)))
end

function update_dt(alg::AbstractFVAlgorithm,u,Flux,
    CFL,mesh)
  maxρ = zero(eltype(u))
  dx = cell_volume(mesh, 1)
  for i in cell_indices(mesh)
    maxρ = max(maxρ, fluxρ(value_at_cell(u,i,mesh), Flux))
  end
  CFL/(1/dx*maxρ)
end

function maxfluxρ(u, Flux, mesh)
    maxρ = zero(eltype(u))
    for i in cell_indices(mesh)
      maxρ = max(maxρ, fluxρ(value_at_cell(u,i,mesh), Flux))
    end
    maxρ
end
