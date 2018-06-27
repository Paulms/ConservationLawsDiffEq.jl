# Copyright (c) 2017: Hendrik Ranocha.

using Parameters
using RecipesBase

"""
An abstract type representing a balance law in `Dim` space dimensions.
"""
abstract type AbstractBalanceLaw{Dim} end

"""
    Burgers{T,Dim}
Burgers' equation
``
  \\partial_t u + \\partial_x \\frac{u^2}{2} = 0
``
in `Dim` space dimensions using `T` as scalar type.
"""
struct Burgers{T,Dim} <: AbstractBalanceLaw{Dim} end

function Burgers(T=Float64, Dim=1)
  Burgers{T,Dim}()
end

"""
    RiemannProblem{Model<:AbstractBalanceLaw{1}, U, T<:Real}
A Riemann problem in one space dimension for `model` with left and right values
`uₗ`,`uᵣ` at `x₀`,`t₀`.
"""
struct RiemannProblem{Model<:AbstractBalanceLaw{1}, U, T<:Real}
  model::Model
  uₗ::U
  uᵣ::U
  x₀::T
  t₀::T
end

"""
    RiemannProblem(model::AbstractBalanceLaw{1}, uₗ, uᵣ, x₀::Real=0, t₀::Real=0)
Create the `RiemannProblem` for `model` with left and right values  `uₗ`,`uᵣ` at
`x₀`,`t₀`.
"""
function RiemannProblem(model::AbstractBalanceLaw{1}, uₗ, uᵣ, x₀::Real=0, t₀::Real=0)
  assert(typeof(uₗ) == typeof(uᵣ))
  x₀, t₀ == promote(x₀, t₀)
  RiemannProblem{typeof(model),typeof(uₗ),typeof(x₀)}(model, uₗ, uᵣ, x₀, t₀)
end

"""
    BurgersRiemannSolution{T,T1}
The solution of a Riemann problem `prob` for Burgers' equation.
"""
struct BurgersRiemannSolution{T,T1} <: Function
  prob::RiemannProblem{Burgers{T,1},T,T1}
  σ⁻::T
  σ⁺::T
end


"""
    minmax_speeds(sol::BurgersRiemannSolution)
Return the minimal and maximal speeds `σ⁻, σ⁺` that appear in the solution `sol`.
"""
function minmax_speeds(sol::BurgersRiemannSolution)
  sol.σ⁻, sol.σ⁺
end


"""
    (sol::BurgersRiemannSolution)(ξ::Real)
Evaluate the solution `sol` at the value `ξ` of the self-similarity variable
`ξ = (x - x₀) / (t - t₀)`.
"""
function (sol::BurgersRiemannSolution)(ξ::Real)
  σ⁻ = sol.σ⁻; σ⁺ = sol.σ⁺
  uₗ = sol.prob.uₗ; uᵣ = sol.prob.uᵣ

  if ξ < σ⁻
    uₗ
  elseif ξ < σ⁺
    uₗ + (ξ-σ⁻)/(σ⁺-σ⁻) * (uᵣ-uₗ)
  else
    uᵣ
  end
end


"""
    (sol::BurgersRiemannSolution)(t::Real, x::Real)
Evaluate the solution `sol` at the time and space coordinates `t` and `x`.
"""
function (sol::BurgersRiemannSolution)(x::Real, t::Real)
  x₀ = sol.prob.x₀; t₀ = sol.prob.t₀

  sol((x-x₀)/(t-t₀))
end

"""
    solve{T,T1}(prob::RiemannProblem{Burgers{T,1},T,T1})
Compute the solution of the Riemann prolem `prob`.
"""
function get_solution(prob::RiemannProblem{Burgers{T,1},T,T1}) where {T,T1}
  uₗ = prob.uₗ; uᵣ = prob.uᵣ
  if uₗ > uᵣ
    σ⁻ = σ⁺ = (uₗ + uᵣ) / 2
  else
    σ⁻ = uₗ
    σ⁺ = uᵣ
  end
  BurgersRiemannSolution(prob, σ⁻, σ⁺)
end
