# 1D Burgers Equation
# u_t+(0.5*u²)_{x}=0

using ConservationLawsDiffEq

const CFL = 0.5
const Tend = 1.0

f(::Type{Val{:jac}},u::Vector) = diagm(u)
f(u::Vector) = u.^2/2

function u0_func(xx)
  N = size(xx,1)
  uinit = zeros(N, 1)
  uinit[:,1] = sin.(2*π*xx)
  return uinit
end

function get_problem(N)
  mesh = Uniform1DFVMesh(N,0.0,1.0,:PERIODIC,:PERIODIC)
  u0 = u0_func(cell_centers(mesh))
  ConservationLawsProblem(u0,f,CFL,Tend,mesh)
end
#Compile
prob = get_problem(10)
#Run
prob = get_problem(200)
@time sol = solve(prob, FVSKTAlgorithm();progress=true, use_threads = true, save_everystep = false)
@time sol1 = fast_solve(prob, FVSKTAlgorithm();progress=true)
@time sol2 = solve(prob, LaxFriedrichsAlgorithm();progress=true, save_everystep = false)
@time sol3 = solve(prob, LocalLaxFriedrichsAlgorithm();progress=true, save_everystep = false)
@time sol4 = solve(prob, GlobalLaxFriedrichsAlgorithm();progress=true, save_everystep = false)
#@time sol5 = solve(prob, LaxWendroff2sAlgorithm();progress=true, save_everystep = false)
@time sol5 = solve(prob, FVCompWENOAlgorithm();progress=true, TimeAlgorithm = SSPRK33())
@time sol6 = solve(prob, FVCompMWENOAlgorithm();progress=true, TimeAlgorithm = SSPRK33())
@time sol7 = solve(prob, FVSpecMWENOAlgorithm();progress=true, save_everystep = false)


#Plot
using Plots;
pyplot()
plot(sol,tidx = 1,lab="uo",line=(:dot,2))
plot!(sol,lab="KT u")
plot!(sol2,lab="L-F h")
plot!(sol3,lab="LLF h")
plot!(sol4,lab="GLF h")
plot!(sol5,lab="Comp WENO5 h")
plot!(sol6,lab="Comp MWENO5 h")
plot!(sol7,lab="Spec MWENO5 h")
