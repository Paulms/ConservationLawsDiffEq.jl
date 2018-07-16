#ENO coefficients for uniform mesh
#Based on: Shu C. Essentially Non-Oscillatory and Weighted Essentially
#Non-Oscillatory Schemes for Hyperbolic Conservation Laws
function unif_crj(k::Int)
  if k == 1
      return([1;1])
  end
  crj = zeros(k+1,k)
  for i = 1:(k+1)
    r = i-2
    for j = 1:k
      psum = 0.0
      for m = j:k
        numer = 0.0
        for l = 0:k
          if (l!=m)
            numer_prod = 1.0
            for q = 0:k
              if (q != m && q != l)
                numer_prod = numer_prod*(r-q+1)
              end
            end
            numer = numer + numer_prod
          end
        end
        denom = 1.0
        for l = 0:k
          if (l!=m);denom = denom * (m-l);end
        end
        psum = psum + numer/denom
      end
      crj[i,j] = psum
    end
  end
  crj
end

# Reconstruction types
struct ENO_Reconstruction{kType} <: AbstractReconstruction
    order::kType
    crj::Matrix{Float64}
end
ENO_Reconstruction(order::Int) = ENO_Reconstruction(order, unif_crj(Int((order + 1)/2)))

struct WENO_Reconstruction{kType,eType} <: AbstractReconstruction
    order::kType
    crj::Matrix{Float64}
    ɛ::eType
end
WENO_Reconstruction(order::Int; ɛ = 1e-6) = WENO_Reconstruction(order, unif_crj(Int((order + 1)/2)), ɛ)

struct MWENO_Reconstruction{kType,eType} <: AbstractReconstruction
    order::kType
    crj::Matrix{Float64}
    ɛ::eType
end
MWENO_Reconstruction(order::Int; ɛ = 1e-6) = MWENO_Reconstruction(order, unif_crj(Int((order + 1)/2)), ɛ)

#Eno reconstruction for uniform mesh
function reconstruct(vloc::AbstractVector, dx::Real, eno_rec::ENO_Reconstruction)
  order = eno_rec.order
  k = Int((order + 1)/2)
  crj = eno_rec.crj
  vdiffs = Vector{typeof(vloc)}(undef,0)
  vl = zero(eltype(vloc))
  vr = zero(eltype(vloc))
  N = size(vloc,1)
  if (N != 2*k-1)
    throw("dimension of vloc is not consistent with order $k ENO")
  end
  # Calculation of divided differences
  push!(vdiffs,copy(vloc))
  for i in 2:k
    push!(vdiffs,(vdiffs[i-1][2:end]-vdiffs[i-1][1:end-1])/dx)
  end
  #Calculation of Stencil
  r = 0
  if k < 2;return(vloc[1],vloc[1]);end
  for m = 2:k
    if (abs(vdiffs[m][k-r-1])<abs(vdiffs[m][k-r]))
      r = r + 1
    end
  end
  for j = 0:(k-1)
    vl = vl + vloc[k-r+j]*crj[r+1,j+1]
    vr = vr + vloc[k-r+j]*crj[r+2,j+1]
  end
  return(vl,vr)
end

#Eno reconstruction for unestructured mesh
function reconstruct(vloc::AbstractVector, xloc::AbstractVector, eno_rec::ENO_Reconstruction)
  order = eno_rec.order
  k = Int((order + 1)/2)
  crj = eno_rec.crj
  vdiffs = Vector{typeof(vloc)}(0)
  vl = zero(eltype(vloc))
  vr = zero(eltype(vloc))
  N = size(vloc,1)
  if (N != 2*k-1)
    throw("dimension of vloc is not consistent with order $k ENO")
  end
  # Calculation of divided differences
  push!(vdiffs,copy(vloc))
  for i in 2:k
    push!(vdiffs,(vdiffs[i-1][2:end]-vdiffs[i-1][1:end-1])./(xloc[i:N]-xloc[1:N-i+1]))
  end
  #Calculation of Stencil
  r = 0
  if k < 2;return(vloc[1],vloc[1]);end
  for m = 2:k
    if (abs(vdiffs[m][k-r-1])<abs(vdiffs[m][k-r]))
      r = r + 1
    end
  end
  for j = 0:(k-1)
    vl = vl + vloc[k-r+j]*crj[r+1,j+1]
    vr = vr + vloc[k-r+j]*crj[r+2,j+1]
  end
  return(vl,vr)
end

#WEno reconstruction for uniform mesh
#Order available: 1, 3, 5
# Jiang, Shu, Efficient Implementation of Weighted ENO Schemes
@inline function get_βk(vloc::Vector, k)
  βk = zeros(k); dr = zeros(k);
  if (k==2)
      dr = [2//3,1//3]
      βk = [(vloc[3]-vloc[2])^2, (vloc[2]-vloc[1])^2]
  elseif (k==3)
      dr = [3//10, 3//5, 1//10]
      βk = [13//12*(vloc[3]-2*vloc[4]+vloc[5])^2 + 1//4*(3*vloc[3]-4*vloc[4]+vloc[5])^2,
      13//12*(vloc[2]-2*vloc[3]+vloc[4])^2 + 1//4*(vloc[2]-vloc[4])^2,
      13//12*(vloc[1]-2*vloc[2]+vloc[3])^2 + 1//4*(3*vloc[3]-4*vloc[2]+vloc[1])^2]
  else
    throw("WENO reconstruction of order $order is not implemented yet!")
  end
  return βk, dr
end

function reconstruct(vloc::AbstractVector, rec_scheme::WENO_Reconstruction)
  order = rec_scheme.order
  k = Int((order + 1)/2)
  crj = rec_scheme.crj
  ɛ = rech_scheme.ɛ
  vl = zero(eltype(vloc))
  vr = zero(eltype(vloc))
  N = size(vloc,1)
  if (N != order)
    throw("dimension of vloc is not consistent with order $order WENO")
  end

  #Special case k = 1
  if (k == 1)
    vl = vloc[1]; vr = vloc[1]
    return vl, vr
  end

  # Apply WENO procedure
  αl = zeros(k); αr = zeros(k);
  ωl = zeros(k); ωr = zeros(k);
  βk = zeros(k); dr = zeros(k);

  # Compute k values of xl and xr based on different stencils
  ulr = zeros(k); urr = zeros(k);
  for r=0:(k-1)
      for i=0:k-1
          urr[r+1] = urr[r+1] + crj[r+2,i+1]*vloc[k-r+i];
          ulr[r+1] = ulr[r+1] + crj[r+1,i+1]*vloc[k-r+i];
      end
  end

  # Set up WENO coefficients for different orders - 2k-1
  βk, dr = get_βk(vloc, k)

  # Compute α parameters
  for r=1:k
      αr[r] = dr[r]/(ɛ+βk[r])^2;
      αl[r] = dr[k+1-r]/(ɛ+βk[r])^2;
  end

  # Compute wENO weights parameters
  for r=1:k
      ωl[r] = αl[r]/sum(αl);
      ωr[r] = αr[r]/sum(αr);
  end

  # Compute cell interface values
  for r=1:k
      vl = vl + ωl[r]*ulr[r];
      vr = vr + ωr[r]*urr[r];
  end
  return(vl,vr)
end

function reconstruct(vmloc::AbstractVector, vploc::AbstractVector,rec_scheme::WENO_Reconstruction)
  order = rec_scheme.order
  k = Int((order + 1)/2)
  crj = rec_scheme.crj
  ɛ = rec_scheme.ɛ
  vl = zero(eltype(vmloc))
  vr = zero(eltype(vploc))
  N = size(vmloc,1)
  if (N != order)
    throw("dimension of vloc is not consistent with order $order WENO")
  end
  #Special case k = 1
  if (k == 1)
    vl = vmloc[1]; vr = vploc[1]
    return vl, vr
  end

  # Apply WENO procedure
  αl = zeros(k); αr = zeros(k);
  ωl = zeros(k); ωr = zeros(k);
  βk = zeros(k); dr = zeros(k);

  # Compute k values of xl and xr based on different stencils
  ulr = zeros(k); urr = zeros(k);
  for r=0:(k-1)
      for i=0:k-1
          urr[r+1] = urr[r+1] + crj[r+2,i+1]*vploc[k-r+i];
          ulr[r+1] = ulr[r+1] + crj[r+1,i+1]*vmloc[k-r+i];
      end
  end

  # Set up WENO coefficients for different orders - 2k-1
  βrk, dr = get_βk(vploc, k)
  βlk, dr = get_βk(vmloc, k)

  # Compute α parameters
  for r=1:k
      αr[r] = dr[r]/(ɛ+βrk[r])^2;
      αl[r] = dr[k+1-r]/(ɛ+βlk[r])^2;
  end

  # Compute wENO weights parameters
  for r=1:k
      ωl[r] = αl[r]/sum(αl);
      ωr[r] = αr[r]/sum(αr);
  end

  # Compute cell interface values
  for r=1:k
      vl = vl + ωl[r]*ulr[r];
      vr = vr + ωr[r]*urr[r];
  end
  return(vl,vr)
end


#Mapped WEno reconstruction for uniform mesh
#Reference:
# A. Henrick, T. Aslam, J. Powers, Mapped weighted essentially non-oscillatory
# schemes: Achiving optimal order near critical points

#Mapping function
@inline function gk(ω::Vector, dr::Vector)
  g = zero(ω)
  for i in 1:size(ω,1)
    g[i] = ω[i]*(dr[i]+dr[i]^2-3*dr[i]*ω[i]+ω[i]^2)/(dr[i]^2+ω[i]*(1-2*dr[i]))
  end
  g
end

function reconstruct(vloc::AbstractVector, rec_scheme::MWENO_Reconstruction)
  order = rec_scheme.order
  k = Int((order + 1)/2)
  crj = rec_scheme.crj
  ɛ = rech_scheme.ɛ
  vl = zero(eltype(vloc))
  vr = zero(eltype(vloc))
  N = size(vloc,1)
  if (N != order)
    throw("dimension of vloc is not consistent with order $order WENO")
  end
  #Special case k = 1
  if (k == 1)
    vl = vloc[1]; vr = vloc[1]
    return vl, vr
  end

  # Apply WENO procedure
  αl = zeros(k); αr = zeros(k);
  ωl = zeros(k); ωr = zeros(k);
  αml = zeros(k); αmr = zeros(k);
  ωml = zeros(k); ωmr = zeros(k);
  βk = zeros(k); dr = zeros(k);

  # Compute k values of xl and xr based on different stencils
  ulr = zeros(k); urr = zeros(k);
  for r=0:(k-1)
      for i=0:k-1
          urr[r+1] = urr[r+1] + crj[r+2,i+1]*vloc[k-r+i];
          ulr[r+1] = ulr[r+1] + crj[r+1,i+1]*vloc[k-r+i];
      end
  end

  # Set up WENO coefficients for different orders - 2k-1
  if (k==2)
      dr = [2//3,1//3]
      βk = [(vloc[3]-vloc[2])^2, (vloc[2]-vloc[1])^2]
  elseif (k==3)
      dr = [3//10, 3//5, 1//10]
      βk = [13//12*(vloc[3]-2*vloc[4]+vloc[5])^2 + 1//4*(3*vloc[3]-4*vloc[4]+vloc[5])^2,
      13//12*(vloc[2]-2*vloc[3]+vloc[4])^2 + 1//4*(vloc[2]-vloc[4])^2,
      13//12*(vloc[1]-2*vloc[2]+vloc[3])^2 + 1//4*(3*vloc[3]-4*vloc[2]+vloc[1])^2]
  else
    throw("WENO reconstruction of order $order is not implemented yet!")
  end

  # Compute α parameters
  for r=1:k
      αr[r] = dr[r]/(ɛ+βk[r])^2;
      αl[r] = dr[k+1-r]/(ɛ+βk[r])^2;
  end

  # Compute wENO weights parameters
  for r=1:k
      ωl[r] = αl[r]/sum(αl);
      ωr[r] = αr[r]/sum(αr);
  end

  # Compute α mapped parameters
  αmr = gk(ωr)
  αml = gk(ωl)

  # Compute mapped wENO weights parameters
  for r=1:k
      ωml[r] = αml[r]/sum(αml);
      ωmr[r] = αmr[r]/sum(αmr);
  end

  # Compute cell interface values
  for r=1:k
      vl = vl + ωml[r]*ulr[r];
      vr = vr + ωmr[r]*urr[r];
  end
  return(vl,vr)
end

function reconstruct(vmloc::AbstractVector, vploc::AbstractVector,rec_scheme::MWENO_Reconstruction)
  order = rec_scheme.order
  k = Int((order + 1)/2)
  crj = rec_scheme.crj
  ɛ = rec_scheme.ɛ
  vl = zero(eltype(vmloc))
  vr = zero(eltype(vploc))
  N = size(vmloc,1)
  if (N != order)
    throw("dimension of vloc is not consistent with order $order WENO")
  end
  #Special case k = 1
  if (k == 1)
    vl = vmloc[1]; vr = vploc[1]
    return vl, vr
  end

  # Apply WENO procedure
  αl = zeros(k); αr = zeros(k);
  ωl = zeros(k); ωr = zeros(k);
  αml = zeros(k); αmr = zeros(k);
  ωml = zeros(k); ωmr = zeros(k);
  βk = zeros(k); dr = zeros(k);

  # Compute k values of xl and xr based on different stencils
  ulr = zeros(k); urr = zeros(k);
  for r=0:(k-1)
      for i=0:k-1
          urr[r+1] = urr[r+1] + crj[r+2,i+1]*vploc[k-r+i];
          ulr[r+1] = ulr[r+1] + crj[r+1,i+1]*vmloc[k-r+i];
      end
  end

  # Set up WENO coefficients for different orders - 2k-1
  βrk, dr = get_βk(vploc, k)
  βlk, dr = get_βk(vmloc, k)

  # Compute α parameters
  for r=1:k
      αr[r] = dr[r]/(ɛ+βrk[r])^2;
      αl[r] = dr[k+1-r]/(ɛ+βlk[r])^2;
  end

  # Compute wENO weights parameters
  for r=1:k
      ωl[r] = αl[r]/sum(αl);
      ωr[r] = αr[r]/sum(αr);
  end

  # Compute α mapped parameters
  drl = reverse(dr)
  αmr = gk(ωr,dr)
  αml = gk(ωl,drl)

  # Compute mapped wENO weights parameters
  for r=1:k
      ωml[r] = αml[r]/sum(αml);
      ωmr[r] = αmr[r]/sum(αmr);
  end

  # Compute cell interface values
  for r=1:k
      vl = vl + ωml[r]*ulr[r];
      vr = vr + ωmr[r]*urr[r];
  end
  return(vl,vr)
end
