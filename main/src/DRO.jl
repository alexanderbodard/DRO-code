###
# Main file for the eventual project
###
include("functions/proj_FT.jl")
using ProximalOperators, LinearAlgebra, Random
Random.seed!(1234)

## Problem statement

N = 20
T = 10
A = [Matrix(1.0I, N, N) for _ in 1:T-1]  #rand(N, N)
A = map((x -> x' * x), A)
b = [zeros(N) for _ in 1:T-1]
x = rand(N)
f = ProximalOperators.Quadratic(A[1], b[1])
gamma = 0.2
s = rand()*f(x) # TODO: Fix edge case where rand() returns 0

## Project onto F_T
@time begin
    for _ in 1:100
        local g_lb = 1e-4 # TODO: How close to zero?
        local g_ub = 1.
        local tol = 1e-4

        proj_FT!(x, s, g_lb, g_ub, tol, f)
    end
end