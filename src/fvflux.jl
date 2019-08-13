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
