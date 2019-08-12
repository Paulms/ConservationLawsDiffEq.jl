struct fvflux{ftype,dftype}
    flux::ftype
    Dflux::dftype
end

function flux_function(f, Df)
    if Df == nothing
        Df = x -> ForwardDiff.jacobian(f,x)
    end
    fvflux(f,Df)
end

@inline evalJacobian(f::fvflux, uj) = fvflux.Dflux(uj)
