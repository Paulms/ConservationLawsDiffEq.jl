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

mutable struct CLFunction{F1,F2}
  f::F1
  Jf::F2
end

(f::CLFunction)(::Type{Val{:jac}}, args...) = f.Jf(args...)
(f::CLFunction)(args...) = f.f(args...)
has_jac(f::CLFunction) = f.Jf != nothing

isinplace(prob::AbstractConservationLawProblem{islinear,isstochastic,MeshType}) where {islinear,isstochastic,MeshType} = false

function ConservationLawsProblem(f0,f,CFL,tend,mesh; jac = nothing)
 if typeof(tend) <: Int
     warn("Integer time passed. It could result in unpredictable behaviour consider using a rational time")
 end
 if jac == nothing
  jac = x -> ForwardDiff.jacobian(f,x)
 end
 numvars = size(f0(cell_faces(mesh)[1]),1)
 islinear = false
 isstochastic = false
 ff = CLFunction(f,jac)
 ConservationLawsProblem{islinear,isstochastic,typeof(mesh),typeof(ff),typeof(CFL),typeof(tend),typeof(f0)}(f0,ff,CFL,(0.0,tend),numvars,mesh)
end

function ConservationLawsWithDiffusionProblem(f0,f,BB,CFL,tend,mesh; jac = nothing)
  if typeof(tend) <: Int
    warn("Integer time passed. It could result in unpredictable behaviour consider using a rational time")
  end
  if jac == nothing
    jac = x -> ForwardDiff.jacobian(f,x)
  end
 numvars = size(f0(cell_faces(mesh)[1]),1)
 islinear = false
 isstochastic = false
 ff = CLFunction(f,jac)
 ConservationLawsWithDiffusionProblem{islinear,isstochastic,typeof(mesh),typeof(ff),typeof(CFL),typeof(tend),typeof(f0),typeof(BB)}(f0,ff,CFL,(0.0,tend),numvars,mesh,BB)
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
