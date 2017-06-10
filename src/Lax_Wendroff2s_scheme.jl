# Ritchmeyer Two-step Lax-Wendroff Method
# Reference:
# R. Leveque. Finite Volume Methods for Hyperbolic Problems.Cambridge University
# Press. New York 2002

immutable LaxWendroff2sAlgorithm <: AbstractFVAlgorithm end

# Numerical Fluxes
#   1   2   3          N-1  N
# |---|---|---|......|---|---|
# 1   2   3   4 ... N-1  N  N+1

function FV_solve{tType,uType,tAlgType,F}(integrator::FVIntegrator{LaxWendroff2sAlgorithm,
  Uniform1DFVMesh,tType,uType,tAlgType,F};kwargs...)
  @fv_deterministicpreamble
  @fv_uniform1Dmeshpreamble
  @fv_generalpreamble
  function nflux(ul, ur,dx,dt)
    Flux(0.5*(ul+ur)-dx/(2*dt)*(Flux(ur)-Flux(ul)))
  end
  @fv_method_with_nflux_common
end
