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

function dg_norm(u, basis::PolynomialBasis, p)
    @assert p > 0.0 "p must be a positive number"
    if p == Inf
        maximum(abs.(u))
    else
        k =  basis.order + 1
        N = size(u,1)
        sum([(abs.(u[i:(i+k-1),:]).^p)'*basis.weights for i in 1:k:N])[1]
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
function get_LP_error(ref::Function, sol::AbstractFVSolution; relative::Bool = true, p = 1.0, pointwise::Bool = false)
    x = cell_centers(sol.prob.mesh)
    tspan = sol.prob.tspan
    uexact = fill!(similar(sol.u[end]), zero(eltype(sol.u[end])))
    faces = cell_faces(sol.prob.mesh)
    for i in cell_indices(sol.prob.mesh)
        if pointwise
            uexact[i,:] = ref(x[i], tspan[end])
        else
            uexact[i,:] = num_integrate(x->ref(x, tspan[end]),faces[i], faces[i+1])/cell_volume(i, sol.prob.mesh)
        end
    end
    relative ? 100*mesh_norm((sol.u[end] - uexact), sol.prob.mesh, p)/mesh_norm(uexact, sol.prob.mesh, p) : mesh_norm((sol.u[end] - uexact), sol.prob.mesh, p)
end

function get_LP_error(ref::Function, sol::DGSolution; relative::Bool = true, p = 1.0)
    x = sol.nodes
    tspan = sol.prob.tspan
    uexact = fill!(similar(sol.u[end]), zero(eltype(sol.u[end])))
    for i in 1:size(x,1)
        uexact[i,:] .= ref(x[i], tspan[end])
    end
    relative ? 100*dg_norm((sol.u[end] - uexact), sol.basis, p)/dg_norm(uexact, sol.basis, p) : dg_norm((sol.u[end] - uexact), sol.basis, p)
end

function get_L1_error(ref::Function, sol::AbstractFVSolution)
    get_LP_error(ref, sol; relative = false)
end

function get_relative_L1_error(ref::Function, sol::AbstractFVSolution)
    get_LP_error(ref, sol)
end

function get_L1_error(ref::Function, sol::DGSolution)
    get_LP_error(ref, sol; relative = false)
end

function get_relative_L1_error(ref::Function, sol::DGSolution)
    get_LP_error(ref, sol)
end

#TODO: Estimate approx numerical errors when DG is used
"Compute aproximate L1 errors with numerical reference solution"
function get_num_LP_error(reference,M, uu,N,dx; relative::Bool = true, p = 1.0)
    uexact = fill!(similar(uu), zero(eltype(uu)))
    R = Int(round(M/N))
    for i = 1:N
        uexact[i,:] = 1.0/R*sum(reference[R*(i-1)+1:R*i,:],dims=1)
    end
    relative ? 100.0*mesh_norm((uu - uexact), dx, p)/mesh_norm(uexact, dx, p) : mesh_norm((uu - uexact), dx, p)
end

function approx_L1_error(sol_ref::AbstractFVSolution, sol::AbstractFVSolution)
    M = numcells(sol_ref.prob.mesh)
    N = numcells(sol.prob.mesh)
    return get_num_LP_error(sol_ref.u[end],M, sol.u[end],N,sol.prob.mesh.Δx;relative = false)
end

function approx_relative_L1_error(sol_ref::AbstractFVSolution, sol::AbstractFVSolution)
    M = numcells(sol_ref.prob.mesh)
    N = numcells(sol.prob.mesh)
    return get_num_LP_error(sol_ref.u[end],M, sol.u[end],N,sol.prob.mesh.Δx;relative = true)
end

# function estimate_error_cubic(reference,M, xx,uu,N)
#   uexact = zeros(N)
#   itp = interpolate(reference[:,2], BSpline(Cubic(Flat())),OnCell())
#   i = (M-1)/(reference[M,1]-reference[1,1])*(xx - reference[1,1])+1
#   uexact = itp[i]
#   sum(1.0/N*abs(uu - uexact))
# end

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

Base.show(io::IO, mime, f::FVOOCTable) = show(io, mime, getMDOCCTable(f))
Base.show(io::IO, mime::MIME"text/plain", f::FVOOCTable) = show(io, mime, getMDOCCTable(f))
Base.showable(mime::MIME, f::FVOOCTable) = showable(mime, Markdown.md"")

"""
get_conv_order_table(alg, get_problem, u_exact, mesh_ncells; relative = true, kwargs...)
Compute a table of errors and approximate order of convergence for numerical scheme `alg`
by solving a Conservations Laws problem in a sequence of meshes of cell size given by
`mesh_ncells`.
`get_problem` is a function that returns a ConservationLawsProblem given a number of cells `N`
`kwargs` extra arguments are passed to `solve` function
"""
function get_conv_order_table(alg, get_problem, u_exact::Function, mesh_ncells; relative::Bool = true, kwargs...)
    errors = fill(zero(Float64),size(mesh_ncells,1),2)
    @assert size(mesh_ncells,1) > 2 "mesh_sizes must have at least two elements"
    for (i,N) in enumerate(mesh_ncells)
        prob = get_problem(N)
        sol = solve(prob, alg;kwargs...);
        errors[i,1] = relative ? get_relative_L1_error(u_exact, sol) : get_L1_error(u_exact, sol)
    end
    @. errors[2:end,2] = -log(errors[1:(end-1),1]/errors[2:end,1])/log(mesh_ncells[1:(end-1)]/mesh_ncells[2:end]);
    return FVOOCTable(mesh_ncells,errors[:,1],errors[:,2],scheme_short_name(alg))
end

function get_conv_order_table(alg, get_problem, u_exact::AbstractFVSolution, mesh_ncells; relative::Bool = true, kwargs...)
    errors = fill(zero(Float64),size(mesh_ncells,1),2)
    @assert size(mesh_ncells,1) > 2 "mesh_sizes must have at least two elements"
    for (i,N) in enumerate(mesh_ncells)
        prob = get_problem(N)
        sol = solve(prob, alg;kwargs...);
        errors[i,1] = relative ? approx_relative_L1_error(u_exact, sol) : approx_L1_error(u_exact, sol)
    end
    @. errors[2:end,2] = -log(errors[1:(end-1),1]/errors[2:end,1])/log(mesh_ncells[1:(end-1)]/mesh_ncells[2:end]);
    return FVOOCTable(mesh_ncells,errors[:,1],errors[:,2],scheme_short_name(alg))
end
