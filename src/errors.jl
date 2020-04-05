#Compute approximate errors at tend with analytical solution
function mesh_norm(u, mesh::Uniform1DFVMesh, p)
    mesh_norm(u, mesh.Δx, p)
end
function mesh_norm(u, dx::Real, p)
    @assert p > 0.0 "p must be a positive number"
    if p == Inf
        maximum(abs.(u))
    else
        (sum(abs,(u).^p)*dx)^(1/p)
    end
end

"""
function get_LP_error(ref::Function, sol::AbstractFVSolution; relative = true, p = 1.0, pointwise::Bool = false)
    Compute Lp errores of FVSolution `sol` against reference solution `ref`.
    Additional options:
    `relative`: default true. Compute relative or absolute error
    `p`: parameter to define Lp norm
    `poitwise` default false. Compute errors against pointwise values of reference solution or cell averages
"""
function get_LP_error(ref::Function, sol::AbstractFVSolution{T}; relative::Bool = true, p = 1.0, pointwise::Bool = false) where {T}
    x = cell_centers(getmesh(sol))
    #tspan = sol.prob.tspan
    Tend = gettimes(sol)[end]
    uexact = fill!(similar(getvalues(sol)[end]), zero(T))
    faces = cell_facets(getmesh(sol))
    _compute_exact_sol!(uexact, Tend, sol, x, faces, ref, pointwise)
    relative ? 100*mesh_norm((getvalues(sol)[end] - uexact), getmesh(sol), p)/mesh_norm(uexact, getmesh(sol), p) : mesh_norm((getvalues(sol)[end] - uexact), getmesh(sol), p)
end

function _compute_exact_sol!(uexact::AbstractArray{T,2},Tend, sol, x, faces, ref, pointwise) where {T}
    for i in 1:getncells(getmesh(sol))
        if pointwise
            uexact[:,i] = ref(x[i], Tend)
        else
            uexact[:,i] = num_integrate(x->ref(x, Tend),faces[i], faces[i+1])/cell_volume(getmesh(sol), i)
        end
    end
end

function _compute_exact_sol!(uexact::AbstractArray{T,1},Tend, sol, x, faces, ref, pointwise) where {T}
    for i in 1:getncells(getmesh(sol))
        if pointwise
            uexact[i] = ref(x[i], Tend)
        else
            uexact[i] = num_integrate(x->ref(x, Tend),faces[i], faces[i+1])/cell_volume(getmesh(sol), i)
        end
    end
end

function get_L1_error(ref::Function, sol::AbstractFVSolution)
    get_LP_error(ref, sol; relative = false)
end

function get_relative_L1_error(ref::Function, sol::AbstractFVSolution)
    get_LP_error(ref, sol)
end

# "Compute aproximate L1 errors with numerical reference solution"
function get_num_LP_error(reference,M, uu,N,dx; relative::Bool = true, p = 1.0)
    uexact = fill!(similar(uu), zero(eltype(uu)))
    R = Int(round(M/N))
    for i = 1:N
        uexact[:,i] = 1.0/R*sum(reference[:,R*(i-1)+1:R*i],dims=1)
    end
    relative ? 100.0*mesh_norm((uu - uexact), dx, p)/mesh_norm(uexact, dx, p) : mesh_norm((uu - uexact), dx, p)
end

function approx_L1_error(sol_ref::AbstractFVSolution, sol::AbstractFVSolution)
    M = getncells(getmesh(sol_ref))
    N = getncells(getmesh(sol))
    dx = cell_volume(getmesh(sol), 1)
    return get_num_LP_error(getvalues(sol_ref)[end],M, getvalues(sol)[end],N,dx;relative = false)
end

function approx_relative_L1_error(sol_ref::AbstractFVSolution, sol::AbstractFVSolution)
    M = getncells(getmesh(sol_ref))
    N = getncells(getmesh(sol))
    dx = cell_volume(getmesh(sol), 1)
    return get_num_LP_error(getvalues(sol_ref)[end],M, getvalues(sol)[end],N,dx;relative = true)
end

# # function estimate_error_cubic(reference,M, xx,uu,N)
# #   uexact = zeros(N)
# #   itp = interpolate(reference[:,2], BSpline(Cubic(Flat())),OnCell())
# #   i = (M-1)/(reference[M,1]-reference[1,1])*(xx - reference[1,1])+1
# #   uexact = itp[i]
# #   sum(1.0/N*abs(uu - uexact))
# # end

## Order of convergence Tables
struct FVOOCTable{htype, etype, otype,ntype}
  h::htype
  errors::etype
  orders::otype
  alg_name::ntype
end


function getMDOCCTable(f::FVOOCTable)
    ooctable = [["M",L"e_{tot}^{M}",L"\theta_{M}"]]
    for i in 1:size(f.h,1)
        push!(ooctable,[string(f.h[i]), string(f.errors[i]), string(f.orders[i])])
    end
    t = Markdown.Table(ooctable, [:l, :c, :c])
    m = Markdown.MD(t)
    return m
end

display(x::FVOOCTable) = display(getMDOCCTable(x))

show(io::IO, ::MIME"text/html", x::FVOOCTable) =
    println(io, Markdown.html(getMDOCCTable(x)))

scheme_short_name(alg::AbstractFVAlgorithm) =  string(typeof(alg))

"""
get_conv_order_table(alg, get_problem, u_exact, mesh_ncells; relative = true, kwargs...)
Compute a table of errors and approximate order of convergence for numerical scheme `alg`
by solving a Conservations Laws problem in a sequence of meshes of cell size given by
`mesh_ncells`.
`get_problem` is a function that returns a ConservationLawsProblem given a number of cells `N`
`kwargs` extra arguments are passed to `solve` function
"""
function get_conv_order_table(alg,solve, get_problem, u_exact::Function, mesh_ncells, TimeIntegrator; relative::Bool = true, kwargs...)
    errors = fill(zero(Float64),size(mesh_ncells,1),2)
    @assert size(mesh_ncells,1) > 2 "mesh_sizes must have at least two elements"
    for (i,N) in enumerate(mesh_ncells)
        prob,mesh,cb,dt = get_problem(N)
        sol_ode = solve(prob, TimeIntegrator;dt = dt, callback = cb, kwargs...);
        sol = fv_solution(sol_ode, mesh)
        errors[i,1] = relative ? get_relative_L1_error(u_exact, sol) : get_L1_error(u_exact, sol)
    end
    @. errors[2:end,2] = -log(errors[1:(end-1),1]/errors[2:end,1])/log(mesh_ncells[1:(end-1)]/mesh_ncells[2:end]);
    return FVOOCTable(mesh_ncells,errors[:,1],errors[:,2],scheme_short_name(alg))
end

function get_conv_order_table(alg,solve, get_problem, u_exact::AbstractFVSolution, mesh_ncells, TimeIntegrator; relative::Bool = true, kwargs...)
    errors = fill(zero(Float64),size(mesh_ncells,1),2)
    @assert size(mesh_ncells,1) > 2 "mesh_sizes must have at least two elements"
    for (i,N) in enumerate(mesh_ncells)
        prob,mesh,cb,dt = get_problem(N)
        sol_ode = solve(prob, TimeIntegrator;dt = dt, callback = cb, kwargs...);
        sol = fv_solution(sol_ode, mesh)
        errors[i,1] = relative ? approx_relative_L1_error(u_exact, sol) : approx_L1_error(u_exact, sol)
    end
    @. errors[2:end,2] = -log(errors[1:(end-1),1]/errors[2:end,1])/log(mesh_ncells[1:(end-1)]/mesh_ncells[2:end]);
    return FVOOCTable(mesh_ncells,errors[:,1],errors[:,2],scheme_short_name(alg))
end
