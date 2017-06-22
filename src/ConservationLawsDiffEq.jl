__precompile__()
module ConservationLawsDiffEq
  using DiffEqPDEBase,DiffEqBase
  using Reexport
  @reexport using OrdinaryDiffEq

  using Parameters, Compat, Juno
  using ForwardDiff, Interpolations, IterativeSolvers
  using RecipesBase

  # Interfaces
  import DiffEqBase: solve, @def, has_jac, LinSolveFactorize, LinearInterpolation

  #Solutions
  @compat abstract type AbstractFVSolution{T,N} <: AbstractTimeseriesSolution{T,N} end
  # Mesh
  @compat abstract type AbstractFVMesh end
  @compat abstract type AbstractUniformFVMesh <: AbstractFVMesh end
  # Problems
  #@compat abstract type PDEProblem <: DEProblem end
  @compat abstract type AbstractConservationLawProblem{islinear,isstochastic,MeshType} <: PDEProblem end
  # algorithms
  @compat abstract type PDEAlgorithm <: DEAlgorithm end
  @compat abstract type AbstractFVAlgorithm <: PDEAlgorithm end

  include("spatial_mesh.jl")
  include("ConservationLawsProblems.jl")
  include("fv_integrators.jl")
  include("algorithms.jl")
  include("solutions.jl")
  include("fv_solve.jl")
  include("errors.jl")
  include("ArrayUtils.jl")
  include("plotRecipe.jl")

  #Algoritms
  include("KT_scheme.jl")
  include("ENO_WENO.jl")
  include("Tecno_scheme.jl")
  include("ESJP_scheme.jl")
  include("WENO_Scheme.jl")
  include("LI_IMEXRK_Schemes.jl")
  include("Lax_Friedrichs_scheme.jl")
  include("Lax_Wendroff2s_scheme.jl")

  export solve
  export Uniform1DFVMesh
  export FVSolution, save_csv
  export ConservationLawsProblem, ConservationLawsWithDiffusionProblem
  export FVKTAlgorithm, FVTecnoAlgorithm, FVESJPAlgorithm
  export FVCompWENOAlgorithm, FVCompMWENOAlgorithm, FVSpecMWENOAlgorithm
  export RKTable, LI_IMEX_RK_Algorithm
  export LaxFriedrichsAlgorithm, LaxWendroff2sAlgorithm
  export get_L1_errors, minmod
end
