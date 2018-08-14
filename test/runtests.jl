using Test

@time begin
    include("test0.jl")
    include("test2.jl")
    include("test4.jl")
    include("test5.jl")
    include("test_aux.jl")
end
