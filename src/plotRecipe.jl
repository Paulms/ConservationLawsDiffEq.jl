@recipe function f(sol::AbstractFVSolution; tidx = size(sol.t,1), uvars=0)
    seriestype  :=  :path
    xguide --> "x"
    yguide --> "u"
    labels = String[]
    for i in 1:size(sol.u[tidx],2)
      push!(labels,"u$i")
    end
    yvector = sol.u[tidx]
    if uvars != 0
      yvector = sol.u[tidx][:,uvars]
      labels = labels[uvars]
    end
    if typeof(labels) <: String
      label --> labels
    else
      label --> reshape(labels,1,length(labels))
    end
    sol.prob.mesh.x, yvector
end
