module ConservationLawsDiffEq
  using DiffEqCallbacks
  using LinearAlgebra
  using SparseArrays
  using TreeViews
  using DelimitedFiles
  import Tensors

  import ForwardDiff
  using RecipesBase, LaTeXStrings
  import FastGaussQuadrature
  using StaticArrays

  # Interfaces
  import DiffEqBase: DiscreteCallback,AbstractODESolution
  import Base: show
  import Markdown

  # Reconstructions
  abstract type AbstractReconstruction end

  # Limiters
  abstract type AbstractSlopeLimiter end

  #Interface functions
  include("mesh.jl")
  include("mesh_generator.jl")
  include("fvmesh.jl")
  include("fvintegrator.jl")
  include("fvSchemesAPI.jl")
  include("fvflux.jl")
  include("timecallbacks.jl")
  include("fvutils.jl")
  include("fvsolve.jl")
  include("clsolution.jl")
  include("uniform1Dmesh.jl")

  #Algoritms
  include("ENO_WENO.jl")
  # include("Tecno_scheme.jl")
    include("schemes/WENO_Scheme.jl")
    include("schemes/Lax_Friedrichs_scheme.jl")
  include("schemes/Lax_Wendroff2s_scheme.jl")
  include("schemes/CU_scheme.jl")
  include("schemes/SKT_scheme.jl")
    include("limiters.jl")

  # Other
  include("errors.jl")
  include("plotRecipe.jl")

  #Exports
  # Schemes API
  export update_flux_value

  # User API
    export getSemiDiscretization
  export getInitialState
  export get_adaptative_callback, getCFLCallback
  export update_dt!
  export Periodic, ZeroFlux, Dirichlet
  export save_csv
  export fv_solution

  # mesh related functions
  export line_mesh, rectangle_mesh
  export LineCell, TriangleCell, RectangleCell
  export getncells, get_coordinates, getnnodes, getnedges, getnfaces
  export PolytopalMesh, getdimension
  export getnodeset, getnodecoords
  export cell_volume, cell_centroid, cell_diameter
  export Uniform1DFVMesh

  # Schemes
  # export FVTecnoScheme
  export FVCompWENOScheme, FVCompMWENOScheme, FVSpecMWENOScheme
  export LaxFriedrichsScheme
  export LaxWendroff2sScheme, LaxWendroffScheme
  export LocalLaxFriedrichsScheme, GlobalLaxFriedrichsScheme
  export FVCUScheme, FVDRCUScheme
  export FVSKTScheme
  export FVDRCU5Scheme

  # scheme utils
  export get_total_u, get_relative_L1_error, get_L1_error#, approx_L1_error, approx_relative, L1_error
  export num_integrate
  #export FVOOCTable, get_conv_order_table, mesh_norm, get_LP_error, get_num_LP_error
  export GeneralizedMinmodLimiter, OsherLimiter, MinmodLimiter, SuperbeeLimiter
end
