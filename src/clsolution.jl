
abstract type AbstractFVSolution{T,N} end

struct FVSolution{T,N,mType} <: AbstractFVSolution{T,N}
  ode_sol::AbstractODESolution{T,N}
  mesh::mType
end

function Base.show(io::IO, A::FVSolution)
  show(io,A.ode_sol)
  show(io, A.mesh)
end

getvalues(sol::FVSolution) = sol.ode_sol.u
gettimes(sol::FVSolution) = sol.ode_sol.t
getmesh(sol::FVSolution) = sol.mesh

function save_csv(sol::FVSolution, file_name::String; idx = -1)
  if !endswith(file_name,".csv")
    file_name = "$file_name.csv"
  end
  if idx == -1
    writedlm(file_name, hcat(cell_centers(getmesh(sol)),getvalues(sol)[end]), ',')
  else
    writedlm(file_name, hcat(cell_centers(getmesh(sol)),getvalues(sol)[idx]), ',')
  end
end

function get_total_u(sol::FVSolution{T}) where {T}
    masa = fill(zero(T),size(getvalues(sol),1))
    for (i,u) in enumerate(getvalues(sol))
        masa[i] = sum(u*cell_volume(getmesh(sol),i))
    end
    return masa
end