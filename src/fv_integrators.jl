immutable FVIntegrator{T1,mType,tType,uType,tAlgType,F,G}
  alg::T1
  mesh::mType
  u0::uType
  Flux::F
  Jf :: G
  CFL :: Number
  M::Int
  TimeAlgorithm::tAlgType
  tend::tType
end

immutable FVDiffIntegrator{T1,mType,tType,uType,tAlgType,F,G,B}
  alg::T1
  mesh::mType
  u0::uType
  Flux::F
  DiffMat::B
  Jf :: G
  CFL :: Real
  M::Int
  TimeAlgorithm::tAlgType
  tend::tType
end
