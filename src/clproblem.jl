abstract type AbstractConservationLawProblem{isscalar,MeshType}

struct ConservationLawsProblem{isscalar,MeshType,F5,F,DF,T,BC} <: AbstractConservationLawProblem{isscalar,MeshType}
 f0::F5
 f::F
 Df::DF
 tspan::T
 bcs::BC
 numvars::Int
 mesh::MeshType
end

struct ConservationLawsWithDiffusionProblem{isscalar,MeshType,F5,F,DF,T,BC,F6} <: AbstractConservationLawProblem{isscalar,MeshType}
 f0::F5
 f::F
 Df::DF
 tspan::T
 bcs::BC
 numvars::Int
 mesh::MeshType
 DiffMat::F6
end

isscalar(prob::AbstractConservationLawProblem{iss}) = iss

function ConservationLawsProblem(f,f0,mesh,bcs; tspan = [0.0,1.0], Df = nothing)
 if eltype(tspan) <: Int
     @warn("Integer time passed. It could result in unpredictable behaviour consider using a rational time")
 end
 numvars = size(f0(getnodecoords(mesh, 1)[1]),1)
 isscalar = !(numvars > 1)
 ConservationLawsProblem{isscalar,typeof(mesh),typeof(f0),typeof(f),typeof(jac),typeof(tspan),typeof(bcs),typeof(numvars)}(f0,f,Df,tspan,bcs,numvars,mesh)
end

function ConservationLawsWithDiffusionProblem(f,BB,f0,mesh,bcs; tspan = [0.0,1.0],Df = nothing)
  if typeof(tend) <: Int
    @warn("Integer time passed. It could result in unpredictable behaviour consider using a rational time")
  end
 numvars = size(f0(getnodecoords(mesh, 1)[1]),1)
 isscalar = !(numvars > 1)
 ConservationLawsWithDiffusionProblem{isscalar,typeof(mesh),typeof(f0),typeof(ff),typeof(jac),typeof(tspan),typeof(bcs),typeof(numvars),typeof(BB)}(f0,ff,Df,tspan,bcs,numvars,mesh,BB)
end

### Displays
Base.summary(prob::AbstractConservationLawProblem{isscalar,mType}) where {islinear, isstochastic, mType} = string("ConservationLawsProblem with mesh type ",mType)

function Base.show(io::IO, A::AbstractConservationLawProblem)
  println(io,summary(A))
  print(io,"timespan: ")
  show(io,A.tspan)
  println(io)
  print(io,"CFL: ")
  show(io, A.CFL)
  println(io)
  print(io,"number of vars: ")
  show(io, A.numvars)
end
