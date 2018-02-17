struct ConservationLawsProblem{islinear,isstochastic,MeshType,F,F3,F4,F5} <: AbstractConservationLawProblem{islinear,isstochastic,MeshType}
 f0::F5
 f::F
 CFL::F3
 tspan::Tuple{F4,F4}
 numvars::Int
 mesh::MeshType
end

struct ConservationLawsWithDiffusionProblem{islinear,isstochastic,MeshType,F,F3,F4,F5,F6} <: AbstractConservationLawProblem{islinear,isstochastic,MeshType}
 f0::F5
 f::F
 CFL::F3
 tspan::Tuple{F4,F4}
 numvars::Int
 mesh::MeshType
 DiffMat::F6
end

isinplace{islinear,isstochastic,MeshType}(prob::AbstractConservationLawProblem{islinear,isstochastic,MeshType}) = false
Base.summary(prob::AbstractConservationLawProblem) = string(typeof(prob)," with uType ",typeof(prob.u0)," and tType ",typeof(prob.tspan[1]),". In-place: ",isinplace(prob))

function ConservationLawsProblem(f0,f,CFL,tend,mesh)
 numvars = size(f0(cell_faces(mesh)[1]),1)
 islinear = false
 isstochastic = false
 ConservationLawsProblem{islinear,isstochastic,typeof(mesh),typeof(f),typeof(CFL),typeof(tend),typeof(f0)}(f0,f,CFL,(0.0,tend),numvars,mesh)
end

function ConservationLawsWithDiffusionProblem(f0,f,BB,CFL,tend,mesh)
 numvars = size(f0(cell_faces(mesh)[1]),1)
 islinear = false
 isstochastic = false
 ConservationLawsWithDiffusionProblem{islinear,isstochastic,typeof(mesh),typeof(f),typeof(CFL),typeof(tend),typeof(f0),typeof(BB)}(f0,f,CFL,(0.0,tend),numvars,mesh,BB)
end

### Displays
Base.summary(prob::AbstractConservationLawProblem{islinear,isstochastic,mType}) where {islinear, isstochastic, mType} = string("ConservationLawsProblem"," with mType ",mType)

function Base.show(io::IO, A::AbstractConservationLawProblem)
  println(io,summary(A))
  print(io,"timespan: ")
  show(io,A.tspan)
  println(io)
  print(io,"num vars: ")
  show(io, A.numvars)
end
