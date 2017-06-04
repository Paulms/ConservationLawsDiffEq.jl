immutable ZeroFluxMatrix{T, S <: AbstractMatrix} <: AbstractMatrix{T}
    a::S
end
ZeroFluxMatrix{T}(a::AbstractMatrix{T}) = ZeroFluxMatrix{T,typeof(a)}(a)
Base.setindex!(A::ZeroFluxMatrix, v, I...) = Base.setindex!(A.a, v, I...)
Base.size(A::ZeroFluxMatrix) = size(A.a)
Base.linearindexing{T<:ZeroFluxMatrix}(::Type{T}) = Base.LinearFast()
Base.similar{T}(A::ZeroFluxMatrix, ::Type{T}) = ZeroFluxMatrix(similar(A.a, T))
Base.similar(A::ZeroFluxMatrix) = similar(A, eltype(A.a))

function Base.getindex(A::ZeroFluxMatrix, I...)
    checkbounds(Bool, A.a, I...) && return A.a[I...]
    if typeof(I[1]) <: Int
      return A.a[min(size(A.a,1),max(1,I[1])),I[2]]
    else
      return A.a[[min(size(A.a,1),max(1,i)) for i in I[1]],I[2]]
    end
end

#Periodic
immutable PeriodicMatrix{T, S <: AbstractMatrix} <: AbstractMatrix{T}
    a::S
end
PeriodicMatrix{T}(a::AbstractMatrix{T}) = PeriodicMatrix{T,typeof(a)}(a)

Base.size(A::PeriodicMatrix) = size(A.a)
Base.linearindexing{T<:PeriodicMatrix}(::Type{T}) = Base.LinearFast()
Base.setindex!(A::PeriodicMatrix, v, I...) = Base.setindex!(A.a, v, I...)
Base.similar{T}(A::PeriodicMatrix, ::Type{T}) = PeriodicMatrix(similar(A.a, T))
Base.similar(A::PeriodicMatrix) = similar(A, eltype(A.a))

function Base.getindex(A::PeriodicMatrix, I...)
    checkbounds(Bool, A.a, I...) && return A.a[I...]
    if typeof(I[1]) <: Int
      return A.a[mod1(I[1], size(A.a,1)),I[2]]
    else
      return A.a[[mod1(i, size(A.a,1)) for i in I[1]],I[2]]
    end
end
