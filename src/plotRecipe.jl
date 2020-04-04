@recipe function f(sol::AbstractFVSolution; tidx = size(gettimes(sol),1), vars=nothing)
    seriestype  :=  :path
    xguide --> "x"
    yguide --> "u"
    labels = String[]
    nvars = getnvars(sol)
    if nvars == 1
      push!(labels,"u")
    else
      for i in 1:nvars
        push!(labels,"u$i")
      end
    end
    yvector = getvalues(sol)[tidx]
    if vars != nothing && nvars > 1
      yvector = getvalues(sol)[tidx][vars,:]
      labels = labels[vars]
    end
    if typeof(labels) <: String
      label --> labels
    else
      label --> reshape(labels,1,length(labels))
    end
    cell_centers(getmesh(sol)), yvector
end

# @recipe function f(sol::DGSolution; tidx = size(sol.t,1), vars=nothing)
#     xguide --> "x"
#     yguide --> "u"
#     labels = String[]
#     for i in 1:size(sol.u[tidx],2)
#       push!(labels,"u$i")
#     end
#     yvector = sol.u[tidx]
#     if vars != nothing
#       yvector = sol.u[tidx][:,vars]
#       labels = labels[vars]
#     end
#     if typeof(labels) <: String
#       label --> labels
#     else
#       label --> reshape(labels,1,length(labels))
#     end
#     @series begin
#       seriestype  :=  :path
#       sol.nodes, yvector
#     end
# end

# @recipe function f(ooc::FVOOCTable)
#     seriestype  :=  :path
#     xguide --> "M"
#     yguide --> "error"
#     xscale --> :log
#     yscale --> :log
#     lab --> ooc.alg_name
#     ooc.h, ooc.errors
# end
