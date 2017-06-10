type ConservationLawsProblem{islinear,isstochastic,MeshType,F,F3,F4,F5} <: AbstractConservationLawProblem{islinear,isstochastic,MeshType}
 u0::F5
 f::F
 CFL::F3
 tspan::Tuple{F4,F4}
 numvars::Int
 mesh::MeshType
end

type ConservationLawsWithDiffusionProblem{islinear,isstochastic,MeshType,F,F3,F4,F5,F6} <: AbstractConservationLawProblem{islinear,isstochastic,MeshType}
 u0::F5
 f::F
 CFL::F3
 tspan::Tuple{F4,F4}
 numvars::Int
 mesh::MeshType
 DiffMat::F6
end

isinplace{islinear,isstochastic,MeshType}(prob::AbstractConservationLawProblem{islinear,isstochastic,MeshType}) = false
Base.summary(prob::AbstractConservationLawProblem) = string(typeof(prob)," with uType ",typeof(prob.u0)," and tType ",typeof(prob.tspan[1]),". In-place: ",isinplace(prob))

function ConservationLawsProblem(u0,f,CFL,tend,mesh)
 numvars = size(u0,2)
 islinear = false
 isstochastic = false
 ConservationLawsProblem{islinear,isstochastic,typeof(mesh),typeof(f),typeof(CFL),typeof(tend),typeof(u0)}(u0,f,CFL,(0.0,tend),numvars,mesh)
end

function ConservationLawsWithDiffusionProblem(u0,f,BB,CFL,tend,mesh)
 numvars = size(u0,2)
 islinear = false
 isstochastic = false
 ConservationLawsWithDiffusionProblem{islinear,isstochastic,typeof(mesh),typeof(f),typeof(CFL),typeof(tend),typeof(u0),typeof(BB)}(u0,f,CFL,(0.0,tend),numvars,mesh,BB)
end
