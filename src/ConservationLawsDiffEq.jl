module ConservationLawsDiffEq
  using DiffEqCallbacks
  using LinearAlgebra
  using SpecialFunctions
  using SparseArrays
  using Logging
  using TreeViews
  using DelimitedFiles

  import ForwardDiff, Interpolations, IterativeSolvers
  using RecipesBase, LaTeXStrings
  import FastGaussQuadrature
  using Polynomials
  using StaticArrays

  # Interfaces
  import DiffEqBase: solve, LinSolveFactorize, LinearInterpolation, AbstractTimeseriesSolution
  import Base: show
  import Markdown

  #DG definitions
  #abstract type AbstractDGLimiter end

  # Reconstructions
  #abstract type AbstractReconstruction end

  # Limiters
  #abstract type AbstractSlopeLimiter end

  #Interface functions
  include("mesh.jl")
  include("mesh_generator.jl")
  include("fvmesh.jl")
  include("fvflux.jl")
  include("fvintegrator.jl")
  include("fvalgorithmsAPI.jl")
  include("timecallbacks.jl")
  include("fvutils.jl")
  include("fvsolve.jl")
  include("clproblem.jl")
  include("clsolution.jl")

  #Algoritms
  # include("ENO_WENO.jl")
  # include("Tecno_scheme.jl")
  # include("ESJP_scheme.jl")
  # include("WENO_Scheme.jl")
  # include("LI_IMEXRK_Schemes.jl")
  include("schemes/Lax_Friedrichs_scheme.jl")
  # include("Lax_Wendroff2s_scheme.jl")
  # include("CU_scheme.jl")
  # include("DRCU_scheme.jl")
  # include("DRCU5_scheme.jl")
  # include("SKT_scheme.jl")
  # include("DG_Basis.jl")
  # include("DiscontinuousGalerkin_scheme.jl")
  # include("NumericalFluxes.jl")
  # include("limiters.jl")

  # Other
  include("errors.jl")
  include("plotRecipe.jl")

  #Exports
  # Schemes API
  export update_flux_value

  # User API
  export solve
  export getSemiDiscretization
  export getInitialState
  export get_adaptative_callback, getCFLCallback
  export update_dt!
  export Periodic, ZeroFlux, Dirichlet
  export save_csv
  export ConservationLawsProblem, ConservationLawsWithDiffusionProblem

  # mesh related functions
  export line_mesh, rectangle_mesh
  export LineCell, TriangleCell, RectangleCell
  export getncells, get_coordinates, getnnodes, getnedges, getnfaces
  export PolytopalMesh, getdimension
  export getnodeset, getnodecoords
  export cell_volume, cell_centroid, cell_diameter

  # Schemes
  # export FVTecnoAlgorithm, FVESJPAlgorithm
  # export FVCompWENOAlgorithm, FVCompMWENOAlgorithm, FVSpecMWENOAlgorithm
  # export RKTable, LI_IMEX_RK_Algorithm
  export LaxFriedrichsAlgorithm
  # export LaxWendroff2sAlgorithm, LaxWendroffAlgorithm
  # export LocalLaxFriedrichsAlgorithm, GlobalLaxFriedrichsAlgorithm
  # export DiscontinuousGalerkinScheme
  # export COMP_GLF_Diff_Algorithm
  # export minmod
  # export FVCUAlgorithm, FVDRCUAlgorithm, FVSKTAlgorithm
  # export FVDRCU5Algorithm

  # scheme utils
  export get_total_u, get_relative_L1_error, get_L1_error, approx_L1_error, approx_relative, L1_error
  export num_integrate
  export FVOOCTable, get_conv_order_table, mesh_norm, get_LP_error, get_num_LP_error
  # export advection_num_flux, rusanov_euler_num_flux, glf_num_flux
  # export legendre_basis, PolynomialBasis
  # export ENO_Reconstruction, WENO_Reconstruction, MWENO_Reconstruction

  #Limiter
  # export DGLimiter, Linear_MUSCL_Limiter, WENO_Limiter
  # export GeneralizedMinmodLimiter, OsherLimiter, MinmodLimiter, SuperbeeLimiter
end
