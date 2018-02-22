"Global Lax Friedrichs Flux"
function glf_num_flux(ul, ur, f, α)
    return 0.5*(f(ul)+f(ur))-α/2*(ur-ul)
end

"Flux for advection equation (simple upwinding)"
advection_num_flux(ul, ur, f, α) = ul

"""
Returns the Rusanov interface flux for the Euler equations
V. V. Rusanov, Calculation of Interaction of Non-Steady Shock Waves with
Obstacles, J. Comput. Math. Phys. USSR, 1, pp. 267-279, 1961.
"""
function rusanov_euler_num_flux(ul, ur, f, α)
  @assert(size(ul,1) == size(ur,1),
  "ul and ur vector have different number of components")
  F = zeros(ul)

  # Fisical variables and sound speeds
  ρL = ul[1];vL=ul[2]/ρL;EL = ul[3]
  pL = (γ - 1)*(EL - 0.5*ρL*vL*vL)
  aL = sqrt(γ*max(pL/ρL,1.0))

  ρR = ur[1];vR=ur[2]/ρR;ER = ur[3]
  pR = (γ - 1)*(ER - 0.5*ρR*vR*vR)
  aR = sqrt(γ*max(pR/ρR,1.0))

  # Find the maximum eigenvalue for each interface
  maxρ = max(abs(vL)+aL, abs(vR)+aR)

  # f₁ = ρ*u
  F[1] = 0.5*(ρL*vL + ρR*vR - maxρ*(ρR - ρL))
  # f₂ = rho*u*u+p
  F[2] = 0.5*(ρL*vL*vL + pL + ρR*vR*vR +
                   pR - maxρ*(ρR*vR - ρL*vL))
  # f₃ = (E+p)*u
  F[3] = 0.5*((EL + pL)*vL + (ER + pR)*vR - maxρ*(ER - EL))
  return F
end
