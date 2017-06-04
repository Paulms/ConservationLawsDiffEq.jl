type ConservationLawsProblem{MeshType,F,F2,F3,F4,F5} <: AbstractConservationLawProblem{MeshType}
 u0::F5
 f::F
 Jf::F2
 CFL::F3
 tend::F4
 numvars::Int
 mesh::MeshType
end

type ConservationLawsWithDiffusionProblem{MeshType,F,F2,F3,F4,F5,F6} <: AbstractConservationLawProblem{MeshType}
 u0::F5
 f::F
 Jf::F2
 CFL::F3
 tend::F4
 numvars::Int
 mesh::MeshType
 DiffMat::F6
end

function ConservationLawsProblem(u0,f,CFL,tend,mesh;Jf=nothing)
 numvars = size(u0,2)
 ConservationLawsProblem{typeof(mesh),typeof(f),typeof(Jf),typeof(CFL),typeof(tend),typeof(u0)}(u0,f,Jf,CFL,tend,numvars,mesh)
end

function ConservationLawsWithDiffusionProblem(u0,f,BB,CFL,tend,mesh;Jf=nothing)
 numvars = size(u0,2)
 ConservationLawsWithDiffusionProblem{typeof(mesh),typeof(f),typeof(Jf),typeof(CFL),typeof(tend),typeof(u0),typeof(BB)}(u0,f,Jf,CFL,tend,numvars,mesh,BB)
end
