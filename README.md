# ConservationLawsDiffEq

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)
[![Build Status](https://travis-ci.org/Paulms/ConservationLawsDiffEq.jl.svg?branch=master)](https://travis-ci.org/Paulms/ConservationLawsDiffEq.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/3x0qjeud3viejfn0?svg=true)](https://ci.appveyor.com/project/Paulms/conservationlawsdiffeq-jl)
[![Coverage Status](https://coveralls.io/repos/Paulms/ConservationLawsDiffEq.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/Paulms/ConservationLawsDiffEq.jl?branch=master)
[![codecov.io](http://codecov.io/github/Paulms/ConservationLawsDiffEq.jl/coverage.svg?branch=master)](http://codecov.io/github/Paulms/ConservationLawsDiffEq.jl?branch=master)

Collection of explicit numerical schemes for solving systems of Conservations Laws (finite volume methods), using method of Lines and an ODE Solver.

Each scheme returns a semidiscretization (discretization in space) that represents a ODE system. Time integration is performed then using [OrdinaryDiffEq](https://github.com/JuliaDiffEq/OrdinaryDiffEq.jl).

The general conservation laws problem is represented by the following PDE,

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial}{\partial&space;t}u&space;&plus;&space;\nabla&space;\cdot&space;f(u)=&space;0,\quad&space;\forall&space;(x,t)\in&space;\mathbb{R}^{n}\times\mathbb{R}_{&plus;}&space;\\&space;u(x,0)&space;=&space;u_{0}(x)\quad&space;\forall&space;x&space;\in&space;\mathbb{R}^{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial}{\partial&space;t}u&space;&plus;&space;\nabla&space;\cdot&space;f(u)=&space;0,\quad&space;\forall&space;(x,t)\in&space;\mathbb{R}^{n}\times\mathbb{R}&space;\\&space;u(x,0)&space;=&space;u_{0}(x)\quad&space;\forall&space;x&space;\in&space;\mathbb{R}^{n}" title="\frac{\partial}{\partial t}u + \nabla \cdot f(u)= 0,\quad \forall (x,t)\in \mathbb{R}^{n}\times\mathbb{R} \\ u(x,0) = u_{0}(x)\quad \forall x \in \mathbb{R}^{n}" /></a>

Solutions follow a conservative finite difference (finite volume) pattern. This method updates cell averages of the solution **u**. For a particular cell *i* it has the general form

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{d}{d&space;t}u(x_i,t)&space;=&space;-&space;\frac{1}{\Delta&space;x_i}(F(x_{i&plus;1/2},t)&space;-&space;F(x_{i-1/2},t))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{d}{d&space;t}u(x_i,t)&space;=&space;-&space;\frac{1}{\Delta&space;x_i}(F(x_{i&plus;1/2},t)&space;-&space;F(x_{i-1/2},t))" title="\frac{d}{d t}u(x_i,t) = - \frac{1}{\Delta x_i}(F(x_{i+1/2},t) - F(x_{i-1/2},t))" /></a>

Where the numerical flux <a href="https://www.codecogs.com/eqnedit.php?latex=F_{i&plus;1/2}(t)&space;=&space;F(u_{i}(t),u_{i&plus;1}(t)))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F_{i&plus;1/2}(t)&space;=&space;F(u_{i}(t),u_{i&plus;1}(t)))" title="F_{i+1/2}(t) = F(u_{i}(t),u_{i+1}(t)))" /></a> is an approximate solution of the Riemann problem at the cell interface <a href="https://www.codecogs.com/eqnedit.php?latex=x_{i&plus;1/2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_{i&plus;1/2}" title="x_{i+1/2}" /></a>.

## Features
### Mesh:
At the moment only a Cartesian 1D uniform mesh is available, using `Uniform1DFVMesh(N,[a,b])` command. Where

`N` = Number of cells

`a,b` = start and end coordinates.


The semidiscretization is obtained by using:

`getSemiDiscretization(f,scheme,mesh,[boundary_conditions]; Df, use_threads,numvars)`

where `f` is the flux function defined as a julia function.

`scheme` is the explicit finite volumes scheme used to discretize the flux (see next section).

`mesh` a valid finite volumes mesh.

`boundary_conditions` a set of boundary conditions among: `Dirichlet()`, `ZeroFlux()` and `Periodic()`.
*Note:* Dirichlet boundary values are defined by the initial condition.

`Df`: is an optional Jacobian of the flux function.

`use_threads`: `true` or `false`.

`numvars`: number of variables of the Conservation Laws system.


### Schemes

The explicit numerical schemes in space currently available are the following:

#### Lax-Friedrichs method

(`LaxFriedrichsScheme()`), Global/Local L-F Scheme (`GlobalLaxFriedrichsScheme()`, `LocalLaxFriedrichsScheme()`), Second order Law-Wendroff Scheme (`LaxWendroffScheme()`), Ritchmeyer Two-step Lax-Wendroff Method (`LaxWendroff2sScheme()`)

* R. LeVeque. Finite Volume Methods for Hyperbolic Problems.Cambridge University Press. New York 2002

#### TECNO Schemes

(`FVTecnoScheme(Nflux;ve, order)`)

* U. Fjordholm, S. Mishra, E. Tadmor, *Arbitrarly high-order accurate entropy stable essentially nonoscillatory schemes for systems of conservation laws*. 2012. SIAM. vol. 50. No 2. pp. 544-573

#### High-Resolution Central Schemes

(`FVSKTScheme(;slopeLimiter=GeneralizedMinmodLimiter())`)

Kurganov, Tadmor, *New High-Resolution Central Schemes for Nonlinear Conservation Laws and Convection–Diffusion Equations*, Journal of Computational Physics, Vol 160, issue 1, 1 May 2000, Pages 241-282

#### Second-Order upwind central scheme

(`FVCUScheme(;slopeLimiter=GeneralizedMinmodLimiter())`)

* Kurganov A., Noelle S., Petrova G., Semidiscrete Central-Upwind schemes for hyperbolic Conservation Laws and Hamilton-Jacobi Equations. SIAM. Sci Comput, Vol 23, No 3m pp 707-740. 2001

#### Dissipation Reduced Central upwind Scheme:

Second-Order (`FVDRCUScheme(;slopeLimiter=GeneralizedMinmodLimiter())`), fifth-order (`FVDRCU5Scheme(;slopeLimiter=GeneralizedMinmodLimiter())`)

* Kurganov A., Lin C., On the reduction of Numerical Dissipation in Central-Upwind # Schemes, Commun. Comput. Phys. Vol 2. No. 1, pp 141-163, Feb 2007.

#### Component Wise Weighted Essentially Non-Oscilaroty (WENO-LF)

(`FVCompWENOScheme(;order)`)

* C.-W. Shu, *High order weighted essentially non-oscillatory schemes for convection dominated problems*, SIAM Review, 51:82-126, (2009).

#### Component Wise Mapped WENO Scheme

(`FVCompMWENOScheme(;order)`)

* A. Henrick, T. Aslam, J. Powers, *Mapped weighted essentially non-oscillatory schemes: Achiving optimal order near critical points*. Journal of Computational Physics. Vol 207. 2005. Pages 542-567


#### Characteristic Wise WENO (Spectral) Scheme

(`FVSpecMWENOScheme(;order)`)

* R. Bürger, R. Donat, P. Mulet, C. Vega, *On the implementation of WENO schemes for a class of polydisperse sedimentation models*. Journal of Computational Physics, Volume 230, Issue 6, 20 March 2011, Pages 2322-2344

*Note:* OrdinaryDiffEq callbacks can be used in order to fix a CFL constant value, or recover the `dt` from adaptative ODE methods in the cases when the finite volumes scheme needs its value (`getCFLCallback` and `get_adaptative_callback` methods, see examples for more information about its use)

*Note:* Limiters available: `GeneralizedMinmodLimiter(;θ=1.0)`, `MinmodLimiter()`, `OsherLimiter(;β=1.0)`, `SuperbeeLimiter()`.

## Example
Scalar Burgers equation:

```julia
# u(x,t)_t+(0.5*u²(x,t))_{x}=0
# u(0,x) = f0(x)

using ConservationLawsDiffEq
using OrdinaryDiffEq
using LinearAlgebra

const CFL = 0.5
# First define the problem data (Jacobian is optional but useful)
Jf(u) = u           #Jacobian
f(u) = u^2/2        #Flux function
f0(x) = sin(2*π*x)  #Initial data distribution

# Now discretize the domain
mesh = Uniform1DFVMesh(10, [0.0, 1.0])

# Now get a explicit semidiscretization (discrete in space) du_h(t)/dt = f_h(u_h(t))
f_h = getSemiDiscretization(f,LaxFriedrichsScheme(),mesh,[Periodic()]; Df = Jf, use_threads = false,numvars = 1)

#Compute discrete initial data
u0 = getInitialState(mesh,f0,use_threads = true)

#Setup ODE problem for a time interval = [0.0,1.0]
ode_prob = ODEProblem(f_h,u0,(0.0,1.0))

#Setup callback in order to fix CFL constant value
cb = getCFLCallback(f_h, CFL)

#Estimate an initial dt
dt = update_dt!(u0, f_h, CFL)

#Solve problem using OrdinaryDiffEq
sol = solve(ode_prob,SSPRK22(); dt = dt, callback = cb)

#Plot solution
#Wrap solution so we can plot using dispatch
u_h = fv_solution(sol, mesh)
using Plots
plot(u_h,tidx = 1,lab="uo",line=(:dot,2)) #Plot initial data
plot!(u_h,lab="LF")                       #Plot LaxFriedrichsScheme solution
```

# Disclamer
** developed for personal use, some of the schemes have not been tested enough!!!**
