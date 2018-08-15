using Test

@time begin
    @time include("test0.jl")
    @time include("test2.jl")
    @time include("test4.jl")
    @time include("test5.jl")
    @time include("test_aux.jl")
end
