# Classic Lax-Friedrichs Method
# Reference:
# R. Leveque. Finite Volume Methods for Hyperbolic Problems.Cambridge University
# Press. New York 2002

immutable LaxFriedrichsAlgorithm <: AbstractFVAlgorithm end

# Numerical Fluxes
#   1   2   3          N-1  N
# |---|---|---|......|---|---|
# 1   2   3   4 ... N-1  N  N+1

function FV_solve{tType,uType,tAlgType,F,G}(integrator::FVIntegrator{LaxFriedrichsAlgorithm,
  Uniform1DFVMesh,tType,uType,tAlgType,F,G};kwargs...)
  @fv_deterministicpreamble
  @fv_uniform1Dmeshpreamble
  @fv_generalpreamble
  @inline function nflux(ul, ur,dx,dt)
    0.5*(Flux(ul)+Flux(ur))-dx/(2*dt)*(ur-ul)
  end
  @fv_method_with_nflux_common
end
