# 1D Shallow water equations with flat bottom
using ConservationLawsDiffEq

const CFL = 0.1
const Tend = 0.2
const gr = 9.8

function f(::Type{Val{:jac}},u::Vector)
  h = u[1]
  q = u[2]
  F =[0.0 1.0;-q^2/h^2+gr*h 2*q/h]
  F
end
f(u::Vector) = [u[2];u[2]^2/u[1]+0.5*gr*u[1]^2]

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

function Nflux(ϕl::Vector, ϕr::Vector)
  hl = ϕl[1]; hr = ϕr[1]
  ul = ϕl[2]/ϕl[1]; ur = ϕr[2]/ϕr[1];
  hm = 0.5*(hl+hr)
  um = 0.5*(ul+ur)
  return([hm*um;hm*um^2+0.5*gr*(0.5*(hl^2+hr^2))])
end
ve(u::Vector) = [gr*u[1]-0.5*(u[2]/u[1])^2;u[2]/u[1]]

function get_problem(N)
  mesh = Uniform1DFVMesh(N,-5.0,5.0,:PERIODIC, :PERIODIC)
  u0 = u0_func(cell_centers(mesh))
  prob = ConservationLawsProblem(u0,f,CFL,Tend,mesh)
end
#Compile
prob = get_problem(10)
#Run
prob = get_problem(200)
@time sol = solve(prob, FVSKTAlgorithm();progress=true)
@time sol2 = solve(prob, FVTecnoAlgorithm(Nflux;ve = ve, order=3);progress=true)
@time sol3 = solve(prob, FVCompWENOAlgorithm();progress=true, TimeAlgorithm = SSPRK33())
@time sol4 = solve(prob, FVCompMWENOAlgorithm();progress=true, TimeAlgorithm = SSPRK33())
@time sol5 = solve(prob, FVSpecMWENOAlgorithm();progress=true, TimeAlgorithm = SSPRK33())

#Plot
using Plots;pyplot();
plot(sol, tidx=1, vars=1, lab="ho",line=(:dot,2))
plot!(sol, vars=1,lab="KT h")
plot!(sol2, vars=1,lab="Tecno h")
plot!(sol3, vars=1,lab="Comp WENO5 h")
plot!(sol4, vars=1,lab="Comp MWENO5 h")
plot!(sol5, vars=1,lab="Spec MWENO5 h")
