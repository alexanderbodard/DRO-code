using ProximalOperators, LinearAlgebra, Random
Random.seed!(1234)

N = 20
T = 10
A = [Matrix(1.0I, N, N) for _ in 1:T-1]  #rand(N, N)
A = map((x -> x' * x), A)

b = [zeros(N) for _ in 1:T-1]
x = rand(N)

f = ProximalOperators.Quadratic(A[1], b[1])

gamma = 0.2
s = rand()*f(x) # TODO: Fix edge case where rand() returns 0

function psi(g)
    p, l_value = ProximalOperators.prox(f, x, g)
    return f(p) - g - s    
end

@time begin
    for _ in 1:100
        local g_lb = 1e-4 # TODO: How close to zero?
        local g_ub = 1.
        local tol = 1e-4

        if psi(g_lb)*psi(g_ub) > 0
            error("Incorrect initial interval. Found $(psi(g_lb)) and $(psi(g_ub))")
        end

        while abs(g_ub-g_lb) > tol
            g_new = (g_lb + g_ub) / 2.
            if psi(g_lb) * psi(g_new) < 0
                g_ub = g_new
            else
                g_lb = g_new
            end
        end
    end
end

# @time begin
#     for _ in 1:1000
#         psi(0.1)
#     end
# end