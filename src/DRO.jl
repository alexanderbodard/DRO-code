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

# ## Project onto F_T
# @time begin
#     for _ in 1:100
#         local g_lb = 1e-4 # TODO: How close to zero?
#         local g_ub = 1.
#         local tol = 1e-4

#         proj_FT!(x, s, g_lb, g_ub, tol, f)
#     end
# end


M = 30
N = 20
x = rand(N)
v = rand(M)
sigma = Matrix(1.0I, M, M)
gamma = Matrix(1.0I, N, N)
lambda = rand()
L = rand(M, N)

function grad_f()
    return 5 * ones(N, 1)
end

function prox_h_c(arg)
    h = ProximalOperators.IndBallLinf(1.0)
    a, b = ProximalOperators.prox(h, arg, sigma)
    return a
end

function eval_L(x)
    return L*x
end

function eval_Lp(x)
    return L'*x
end

while true
    xbar = x - gamma * eval_Lp(v) - gamma * grad_f()
    vbar = prox_h_c(v + sigma * eval_L(2 * xbar - x))
    global x = lambda * xbar + (1 - lambda) * x
    global v = lambda * vbar + (1 - lambda) * v

    println(x)
    sleep(1)
end