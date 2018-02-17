# One dimensional wave equation
using ConservationLawsDiffEq

const CFL = 0.45
const Tend = 1.0
const cc = 1.0

f(::Type{Val{:jac}},u::Vector) = [0.0 cc;cc 0.0]
f(u::Vector) = [0.0 cc;cc 0.0]*u
f0(x) = sin(4*π*x)

Nflux(ϕl::Vector, ϕr::Vector) = 0.5*(f(ϕl)+f(ϕr))
exact_sol(x::Vector, t::Float64) = hcat(0.5*(sin.(4*π*(-t+x))+sin.(4*π*(t+x))),
0.5*(sin.(4*π*(-t+x))-sin.(4*π*(t+x))))

function get_problem(N)
  mesh = Uniform1DFVMesh(N,-1.0,1.0,:PERIODIC, :PERIODIC)
  ConservationLawsProblem(f0,f,CFL,Tend,mesh)
end
#Compile
prob = get_problem(10)
#Run
prob = get_problem(500)

@time sol = solve(prob, FVSKTAlgorithm();progress=true)
@time sol2 = solve(prob, FVTecnoAlgorithm(Nflux;order=3);progress=true)
@time sol3 = solve(prob, FVCompWENOAlgorithm();progress=true)
@time sol4 = solve(prob, FVCompMWENOAlgorithm();progress=true)
@time sol5 = solve(prob, FVSpecMWENOAlgorithm();progress=true)

get_L1_error(sol, exact_sol; nvar = 1) #3.3281897e-3
get_L1_error(sol2, exact_sol; nvar = 1) #1.58081746e-4
get_L1_error(sol3, exact_sol; nvar = 1) #1.08984086e-5
get_L1_error(sol4, exact_sol; nvar = 1) #1.11552929e-5
get_L1_error(sol5, exact_sol; nvar = 1) #1.11552929e-5
#Plot
using Plots;pyplot();
plot(sol, tidx=1, vars=1, lab="yo",line=(:dot,2))
plot!(sol, vars=1, lab="KT y",line = (:dot,2))
plot!(sol2, vars=1,lab="Tecno y",line=(:dot,3))
plot!(sol3, vars=1,lab="Comp WENO y",line=(:dot,3))
plot!(sol4, vars=1,lab="Comp MWENO y",line=(:dot,3))
plot!(sol5, vars=1,lab="Spec MWENO y",line=(:dot,3))
plot!(cell_centers(sol.prob.mesh), exact_sol(cell_centers(sol.prob.mesh),Tend)[:,1],lab="Ref y")
