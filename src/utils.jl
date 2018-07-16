check_first_arg(f,T::Type) = check_first_arg(typeof(f),T)
function check_first_arg(::Type{F}, T::Type) where F
    typ = Tuple{Any, T, Vararg}
    typ2 = Tuple{Any, Type{T}, Vararg} # This one is required for overloaded types
    method_table = Base.MethodList(F.name.mt) # F.name.mt gets the method-table
    for m in method_table
        (m.sig<:typ || m.sig<:typ2) && return true
    end
    return false
end

has_jac(f) = check_first_arg(f, Val{:jac})
