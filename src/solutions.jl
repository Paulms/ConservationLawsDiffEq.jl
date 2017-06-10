#@compat abstract type AbstractFVSolution{T,N} <: DESolution end

type FVSolution{T,N,uType,tType,ProbType,MeshType,IType} <: AbstractFVSolution{T,N}
  u::uType
  t::tType
  prob::ProbType
  dense::Bool
  tslocation::Int
  interp::IType
  retcode::Symbol
end

function FVSolution(u::AbstractArray,t,prob,retcode,interp;dense=true)
  T = eltype(eltype(u))
  N = length((size(u)..., length(u)))
  FVSolution{T,N,typeof(u),typeof(t),typeof(prob),typeof(prob.mesh),typeof(interp)}(u,t,prob,false,0,interp,retcode)
end
(sol::FVSolution)(t,deriv::Type=Val{0};idxs=nothing) = sol.interp(t,idxs,deriv)
(sol::FVSolution)(v,t,deriv::Type=Val{0};idxs=nothing) = sol.interp(v,t,idxs,deriv)

function save_csv(sol::FVSolution, file_name::String; idx = -1)
  if !endswith(file_name,".csv")
    file_name = "$file_name.csv"
  end
  if idx == -1
    writedlm(file_name, hcat(sol.prob.mesh.x,sol.u[end]), ',')
  else
    writedlm(file_name, hcat(sol.prob.mesh.x,sol.u[idx]), ',')
  end
end
