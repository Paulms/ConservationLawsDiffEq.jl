#@compat abstract type AbstractFVSolution{T,N} <: DESolution end

type FVSolution{T,N,uType,tType,ProbType,MeshType} <: AbstractFVSolution{T,N}
  u::uType
  t::tType
  prob::ProbType
  dense::Bool
  tslocation::Int
end

function FVSolution(u::AbstractArray,t,prob)
  T = eltype(eltype(u))
  N = length((size(u)..., length(u)))
  FVSolution{T,N,typeof(u),typeof(t),typeof(prob),typeof(prob.mesh)}(u,t,prob,false,0)
end
