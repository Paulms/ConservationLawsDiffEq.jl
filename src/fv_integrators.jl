struct FVIntegrator{T1,mType,tType,uType,tAlgType,F}
  alg::T1
  mesh::mType
  u0::uType
  Flux::F
  CFL :: Number
  M::Int
  TimeAlgorithm::tAlgType
  tend::tType
end

struct FVDiffIntegrator{T1,mType,tType,uType,tAlgType,F,B}
  alg::T1
  mesh::mType
  u0::uType
  Flux::F
  DiffMat::B
  CFL :: Real
  M::Int
  TimeAlgorithm::tAlgType
  tend::tType
end
