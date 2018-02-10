# ConservationLawsDiffEq

[![Build Status](https://travis-ci.org/Paulms/ConservationLawsDiffEq.jl.svg?branch=master)](https://travis-ci.org/Paulms/ConservationLawsDiffEq.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/3x0qjeud3viejfn0?svg=true)](https://ci.appveyor.com/project/Paulms/conservationlawsdiffeq-jl)
[![Coverage Status](https://coveralls.io/repos/Paulms/ConservationLawsDiffEq.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/Paulms/ConservationLawsDiffEq.jl?branch=master)
[![codecov.io](http://codecov.io/github/Paulms/ConservationLawsDiffEq.jl/coverage.svg?branch=master)](http://codecov.io/github/Paulms/ConservationLawsDiffEq.jl?branch=master)

Collection of numerical schemes for the approximation of Systems of Conservations Laws (finite volume methods). Implementation is influenced by [DifferentialEquations API](http://docs.juliadiffeq.org/latest/).

These PDEs are of the form

<a href="https://www.codecogs.com/eqnedit.php?latex=u_{t}&plus;f(u)_{x}&=0,\qquad\forall(x,t)\in\mathbb{R}^{n}\times\mathbb{R}_{&plus;}\\u(x,0)&=u_{0}(x),\qquad\forall&space;x\in\mathbb{R}^{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u_{t}&plus;f(u)_{x}&=0,\qquad\forall(x,t)\in\mathbb{R}^{n}\times\mathbb{R}_{&plus;}\\u(x,0)&=u_{0}(x),\qquad\forall&space;x\in\mathbb{R}^{n}" title="u_{t}+f(u)_{x}&=0,\qquad\forall(x,t)\in\mathbb{R}^{n}\times\mathbb{R}_{+}\\u(x,0)&=u_{0}(x),\qquad\forall x\in\mathbb{R}^{n}" /></a>

We also consider degenerate convection-diffusion systems (degenerate parabolic-hyperbolic equations) of the form:

<a href="https://www.codecogs.com/eqnedit.php?latex=u_{t}&plus;f(u)_{x}&=(B(u)u_{x})_{x},\qquad\forall(x,t)\in\mathbb{R}^{n}\times\mathbb{R}_{&plus;}\\u(x,0)&=u_{0}(x),\qquad\forall&space;x\in\mathbb{R}^{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u_{t}&plus;f(u)_{x}&=(B(u)u_{x})_{x},\qquad\forall(x,t)\in\mathbb{R}^{n}\times\mathbb{R}_{&plus;}\\u(x,0)&=u_{0}(x),\qquad\forall&space;x\in\mathbb{R}^{n}" title="u_{t}+f(u)_{x}&=(B(u)u_{x})_{x},\qquad\forall(x,t)\in\mathbb{R}^{n}\times\mathbb{R}_{+}\\u(x,0)&=u_{0}(x),\qquad\forall x\in\mathbb{R}^{n}" /></a>

Solutions follow a conservative finite diference (finite volume) pattern. This method updates point values (cell averages) of the solution **u** and has the general form

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{d}{du}u_{i}(t)=-\frac{1}{\Delta_{i}x}(F_{i&plus;1/2}(t)-F_{i-1/2}(t))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{d}{du}u_{i}(t)=-\frac{1}{\Delta_{i}x}(F_{i&plus;1/2}(t)-F_{i-1/2}(t))" title="\frac{d}{du}u_{i}(t)=-\frac{1}{\Delta_{i}x}(F_{i+1/2}(t)-F_{i-1/2}(t))" /></a>

Where the numerical flux <a href="https://www.codecogs.com/eqnedit.php?latex=F_{i&plus;1/2}(t)&space;=&space;F(u_{i}(t),u_{i&plus;1}(t)))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F_{i&plus;1/2}(t)&space;=&space;F(u_{i}(t),u_{i&plus;1}(t)))" title="F_{i+1/2}(t) = F(u_{i}(t),u_{i+1}(t)))" /></a> is an approximate solution of the Riemann problem at the cell interface (x(i+1/2)).

An extra numerical function **P** similar to **F** could be added to account for the Diffusion in the second case.

Time integration of the semi-discrete form is performed using [OrdinaryDiffEq](https://github.com/JuliaDiffEq/OrdinaryDiffEq.jl) algorithms.

## Features
### Mesh:
At the momento only Cartesian 1D uniform mesh available, using `Uniform1DFVMesh(N,a,b,left_boundary, right_boundary)` command. Where

`N` = Number of cells

`a,b` = start and end coordinates.

`left_boundary`,`right_boundary` = boundary type (`:ZERO_FLUX` (default), `:PERIODIC`, `:DIRICHLET` )

*Note:* Dirichlet boundary values are defined by initial condition.

* Problem types: System of Conservation Laws without (`ConservationLawsProblem`) and with degenerate diffusion term (`ConservationLawsWithDiffusionProblem`).

### Algorithms

The algorithms follow the method of lines, so first we compute a semidiscretization in space and time integration is performed using ODE solvers.

* Lax-Friedrichs method (`LaxFriedrichsAlgorithm()`), Global/Local L-F Scheme (`GlobalLaxFriedrichsAlgorithm()`, `LocalLaxFriedrichsAlgorithm()`), Second order Law-Wendroff Scheme (`LaxWendroffAlgorithm()`), Ritchmeyer Two-step Lax-Wendroff Method (`LaxWendroff2sAlgorithm()`)

R. LeVeque. Finite Volume Methods for Hyperbolic Problems.Cambridge University Press. New York 2002

* TECNO Schemes (`FVTecnoAlgorithm(Nflux;ve, order)`)

U. Fjordholm, S. Mishra, E. Tadmor, *Arbitrarly high-order accurate entropy stable essentially nonoscillatory schemes for systems of conservation laws*. 2012. SIAM. vol. 50. No 2. pp. 544-573

* High-Resolution Central Schemes (`FVSKTAlgorithm()`)

Kurganov, Tadmor, *New High-Resolution Central Schemes for Nonlinear Conservation Laws and Convection–Diffusion Equations*, Journal of Computational Physics, Vol 160, issue 1, 1 May 2000, Pages 241-282

* Second-Order upwind central scheme (`FVCUAlgorithm`)

Kurganov A., Noelle S., Petrova G., Semidiscrete Central-Upwind schemes for hyperbolic Conservation Laws and Hamilton-Jacobi Equations. SIAM. Sci Comput, Vol 23, No 3m pp 707-740. 2001

* Dissipation Reduced Central upwind Scheme: Second-Order (`FVDRCUAlgorithm`), fifth-order (`FVDRCU5Algorithm`)

Kurganov A., Lin C., On the reduction of Numerical Dissipation in Central-Upwind # Schemes, Commun. Comput. Phys. Vol 2. No. 1, pp 141-163, Feb 2007.

* Component Wise Weighted Essentially Non-Oscilaroty (WENO-LF) (`FVCompWENOAlgorithm(;order)`)

C.-W. Shu, *High order weighted essentially non-oscillatory schemes for convection dominated problems*, SIAM Review, 51:82-126, (2009).

* Component Wise Mapped WENO Scheme (`FVCompMWENOAlgorithm(;order)`)

A. Henrick, T. Aslam, J. Powers, *Mapped weighted essentially non-oscillatory schemes: Achiving optimal order near critical points*. Journal of Computational Physics. Vol 207. 2005. Pages 542-567

* Component Wise Global Lax-Friedrichs Scheme (`COMP_GLF_Diff_Algorithm()`) (?)

* Characteristic Wise WENO (Spectral) Scheme (`FVSpecMWENOAlgorithm(;order)`)

R. Bürger, R. Donat, P. Mulet, C. Vega, *On the implementation of WENO schemes for a class of polydisperse sedimentation models*. Journal of Computational Physics, Volume 230, Issue 6, 20 March 2011, Pages 2322-2344

* Linearly implicit IMEX Runge-Kutta schemes (`LI_IMEX_RK_Algorithm(;scheme, linsolve)`) (not working...)

(See Time integration methods for RK options (`scheme`), Flux reconstruction uses Comp WENO5, to change linear solver see [DifferentialEquations.jl: Specifying (Non)Linear Solvers](http://docs.juliadiffeq.org/stable/features/linear_nonlinear.html))

S. Boscarino, R. Bürger, P. Mulet, G. Russo, L. Villada, *Linearly implicit IMEX Runge Kutta methods for a class of degenerate convection difussion problems*, SIAM J. Sci. Comput., 37(2), B305–B331

### Time integration methods:

Time integration use OrdinaryDiffEq algorithms (default to `SSPRK22()`)

For IMEX Scheme RK methods: H-CN(2,2,2) `:H_CN_222`, H-DIRK2(2,2,2) `:H_DIRK2_222`, H-LDIRK2(2,2,2) `:H_LDIRK2_222`, H-LDIRK3(2,2,2) `:H_LDIRK3_222`, SSP-LDIRK(3,3,2) `:SSP_LDIRK_332`. For more information see:

* S. Boscarino, P.G. LeFloch and G. Russo. *High order asymptotic-preserving methods for fully nonlinear relaxation problems*. SIAM J. Sci. Comput., 36 (2014), A377–A395.

* S. Boscarino, F. Filbet and G. Russo. *High order semi-implicit schemes for time dependent partial differential equations*. SIAM J. Sci. Comput. September 2016, Volume 68, Issue 3, pp 975–1001

## Example
Hyperbolic Shallow Water system with flat bottom:

```julia

using ConservationLawsDiffEq

const CFL = 0.1
const Tend = 0.2
const gr = 9.8

#Define Optional Jacobian of Flux
function f(::Type{Val{:jac}},u::Vector)
  h = u[1]
  q = u[2]
  F =[0.0 1.0;-q^2/h^2+gr*h 2*q/h]
  F
end

#Flux function:
f(u::Vector) = [u[2];u[2]^2/u[1]+0.5*gr*u[1]^2]

#Initial Condition:
function u0_func(xx)
  N = size(xx,1)
  uinit = zeros(N, 2)
  for i = 1:N
    if xx[i] < 0.0
      uinit[i,1] = 2.0
    else
     uinit[i,1] = 1.0
   end
  end
  return uinit
end

# Setup Mesh
N = 100
mesh = Uniform1DFVMesh(N,-5.0,5.0,:PERIODIC,:PERIODIC)

#Setup initial condition
u0 = u0_func(cell_centers(mesh))

#Setup problem:
prob = ConservationLawsProblem(u0,f,CFL,Tend,mesh)

#Solve problem using Kurganov-Tadmor scheme and Strong Stability Preserving RK33
@time sol = solve(prob, FVSKTAlgorithm();progress=true, TimeIntegrator = SSPRK33())

#Plot
using Plots
plot(sol, tidx=1, vars=1, lab="ho",line=(:dot,2))
plot!(sol, vars=1,lab="KT h")
```

# Disclamer
** developed for personal use, some of the schemes have not been tested enough!!!**
