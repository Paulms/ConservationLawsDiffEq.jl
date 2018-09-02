module ConservationLawsDiffEq
  using DiffEqPDEBase,DiffEqBase
  using DiffEqCallbacks
  using Reexport
  using LinearAlgebra
  using SpecialFunctions
  using SparseArrays
  using Logging
  using TreeViews
  using Markdown


  @reexport using OrdinaryDiffEq

  using ForwardDiff, Interpolations, IterativeSolvers
  using RecipesBase, LaTeXStrings, FastGaussQuadrature
  using Polynomials
  using StaticArrays

  # Interfaces
  import DiffEqBase: solve, @def, LinSolveFactorize, LinearInterpolation, AbstractTimeseriesSolution, DEProblem,
                     DEAlgorithm
  import Base: show
  import Markdown

  #Solutions
  abstract type AbstractFVSolution{T,N} <: AbstractTimeseriesSolution{T,N} end
  # Mesh
  abstract type AbstractFVMesh end

  # Problems
  abstract type AbstractConservationLawProblem{islinear,isstochastic,MeshType} <: DEProblem end
  # abstract algorithms types
  abstract type AbstractFVAlgorithm <: DEAlgorithm end
  abstract type AbstractFEAlgorithm <: DEAlgorithm end
  abstract type AbstractDGLimiter end

  # Reconstructions
  abstract type AbstractReconstruction end

  # Limiters
  abstract type AbstractSlopeLimiter end

  #Interface functions
  include("spatial_mesh.jl")
  include("ConservationLawsProblems.jl")
  include("fv_integrators.jl")
  include("aux_functions.jl")
  include("solutions.jl")
  include("fv_solve.jl")

  #Algoritms
  include("ENO_WENO.jl")
  include("Tecno_scheme.jl")
  include("ESJP_scheme.jl")
  include("WENO_Scheme.jl")
  include("LI_IMEXRK_Schemes.jl")
  include("Lax_Friedrichs_scheme.jl")
  include("Lax_Wendroff2s_scheme.jl")
  include("CU_scheme.jl")
  include("DRCU_scheme.jl")
  include("DRCU5_scheme.jl")
  include("SKT_scheme.jl")
  include("DG_Basis.jl")
  include("DiscontinuousGalerkin_scheme.jl")
  include("NumericalFluxes.jl")
  include("limiters.jl")

  # Other
  include("errors.jl")
  include("plotRecipe.jl")

  #Exports
  export solve, fast_solve, CLFunction
  export AbstractFVAlgorithm
  export Uniform1DFVMesh, AbstractFVMesh1D
  export FVSolution, DGSolution, save_csv
  export ConservationLawsProblem, ConservationLawsWithDiffusionProblem
  export FVTecnoAlgorithm, FVESJPAlgorithm
  export FVCompWENOAlgorithm, FVCompMWENOAlgorithm, FVSpecMWENOAlgorithm
  export RKTable, LI_IMEX_RK_Algorithm
  export LaxFriedrichsAlgorithm, LaxWendroff2sAlgorithm, LaxWendroffAlgorithm
  export LocalLaxFriedrichsAlgorithm, GlobalLaxFriedrichsAlgorithm
  export DiscontinuousGalerkinScheme
  export COMP_GLF_Diff_Algorithm
  export minmod
  export FVCUAlgorithm, FVDRCUAlgorithm, FVSKTAlgorithm
  export FVDRCU5Algorithm
  export cell_faces
  export cell_centers, get_semidiscretization, cell_volume, cell_indices, numcells
  export get_total_u, get_relative_L1_error, get_L1_error, approx_L1_error, approx_relative, L1_error
  export num_integrate
  export FVOOCTable, get_conv_order_table, mesh_norm, get_LP_error, get_num_LP_error
  export advection_num_flux, rusanov_euler_num_flux, glf_num_flux
  export legendre_basis, PolynomialBasis
  export ENO_Reconstruction, WENO_Reconstruction, MWENO_Reconstruction

  #Limiter
  export DGLimiter, Linear_MUSCL_Limiter, WENO_Limiter
  export GeneralizedMinmodLimiter, OsherLimiter, MinmodLimiter, SuperbeeLimiter
end
