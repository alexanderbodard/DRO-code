import LinearAlgebra as LA
import MathOptInterface as MOI
import MathOptSetDistances as MOD

using ProximalOperators, Random, JuMP, MosekTools, SparseArrays, Plots, Profile, DelimitedFiles, ForwardDiff

include("scenario_tree.jl")
include("risk_constraints.jl")
include("dynamics.jl")
include("cost.jl")

include("model.jl")
include("custom_model.jl")
include("dynamics_in_l_vanilla_model.jl")
include("dynamics_in_l_supermann_model.jl")
include("ricatti_vanilla_model.jl")
include("mosek_model.jl")

###
# Problem statement
###

# Scenario tree
N = 2; d = 2; nx = 2; nu = 1
scen_tree = generate_scenario_tree(N, d, nx, nu)

# Dynamics: Based on a discretized car model
T_s = 0.05
A = [[[1.,0.] [T_s, 1.0 - rand()*T_s]] for _ in 1:d]
B = [reshape([0., T_s], :, 1) for _ in 1:d]
dynamics = Dynamics(A, B, nx, nu)

# Cost: Let's take a quadratic cost, equal at all timesteps
Q = LA.Matrix([2.2 0; 0 3.7])
R = reshape([3.2], 1, 1)
cost = Cost(
    collect([
        Q for _ in 1:N
    ]),
    collect([
        R for _ in 1:N
    ])
)

# Risk measures: Risk neutral: A = I, B = [I; -I], b = [1;1;-1;-1]
"""
Risk neutral: A = I, B = [I; -I], b = [0.5;0.5;-0.5;-0.5]
AVaR: A = I, B = [-I, I, 1^T, -1^T], b = [0; p / alpha; 1, -1]
"""
rms = [
    Riskmeasure(
        LA.I(2),
        [LA.I(2); -LA.I(2)],
        [0.5 , 0.5, -0.5, -0.5],
        ConvexCone([MOI.Nonnegatives(2)]),
        ConvexCone([MOI.Nonnegatives(4)])
    ) for _ in 1:scen_tree.n_non_leaf_nodes
]

model = build_model(scen_tree, cost, dynamics, rms, DYNAMICS_IN_L_SOLVER)

"""
Computes prox_f^{gamma}(x) where f is the affine function

f(x) = LA.dot(a, x) + b

with a the k'th unit vector
"""
function prox_f(x, gamma, k)
    res = copy(x)
    res[k] -= gamma
    return res
end

"""
Computes prox_g^{sigma}(x) where is the Fenchal dual of the indicator of some set C

Using the Moreau decomposition this is computed by performing a projection.
"""
function prox_g(model, x, sigma)
    return x - sigma * projection(model, x / sigma)
end

function projection(model, z)
    # 4a
    for ind in model.inds_4a
        z[ind] = MOD.projection_on_set(MOD.DefaultDistance(), z[ind], MOI.Nonpositives(2)) # TODO: Fix polar cone
    end

    # 4b
    z[model.inds_4b] = MOD.projection_on_set(MOD.DefaultDistance(), z[model.inds_4b], MOI.Nonpositives(4)) # TODO: Fix polar cone
    
    # 4c
    for (i, ind) in enumerate(model.inds_4c)
        b_bar = model.b_bars[i]
        dot_p = LA.dot(z[ind], b_bar)
        if dot_p > 0
            z[ind] = z[ind] - dot_p / LA.dot(b_bar, b_bar) .* b_bar
        end
    end

    # 4d: Compute projection
    for (scen_ind, ind) in enumerate(model.inds_4d)
        z_temp = z[ind]
        s = z[ind[end] + 1]

        z[ind], z[ind[end] + 1] = epigraph_bisection(model.Q_bars[scen_ind], z_temp, s)

        # ppp, sss = epigraph_qcqp(LA.diagm(model.Q_bars[scen_ind]), z_temp, s)
        # z[ind], z[ind[end] + 1] = ppp, sss
    end

    # 4e: Dynamics
    z[model.inds_4e] = zeros(length(model.inds_4e))

    # Initial condition
    x0 = [.2, .2]
    z[end - length(x0) + 1 : end] = x0

    return z
end

function prox_f_copy(Q, gamma, x)
    return x ./ (Q .+ 1. / gamma) / gamma
end

function psi_copy(Q, gamma, x, s)
    temp = prox_f_copy(Q, gamma, x)
    return 0.5 * sum(Q .* temp.^2) - gamma - s
end

function bisection_method_copy!(g_lb, g_ub, tol, Q, x, s)
    # if ( psi_copy(Q, g_lb, x, s) + tol ) * ( psi_copy(Q, g_ub, x, s) - tol ) > 0 # only work up to a precision of the tolerance
    #     error("Incorrect initial interval. Found $(psi_copy(Q, g_lb, x, s)) and $(psi_copy(Q, g_ub, x, s)) which results in $(( psi_copy(Q, g_lb, x, s) + tol ) * ( psi_copy(Q, g_ub, x, s) - tol ))")
    # end

    g_new = (g_lb + g_ub) / 2
    ps = psi_copy(Q, g_new, x, s)
    while abs(g_ub - g_lb) > tol #abs(ps) > tol
        if sign(ps) > 0
            g_lb = g_new
        elseif sign(ps) < 0
            g_ub = g_new
        else
            return g_new
            error("Should not happen")
        end
        g_new = (g_lb + g_ub) / 2
        ps = psi_copy(Q, g_new, x, s)
    end
    return g_new
end

function epigraph_bisection(Q, x, t)
    f = 0.5 * sum(Q .* (x.^2))
    if f > t
        local g_lb = 0 # TODO: How close to zero?
        local g_ub = f - t #1. TODO: Can be tighter with gamma
        tol = 1e-12
        gamma_star = bisection_method_copy!(g_lb, g_ub, tol, Q, x, t)
        res = prox_f_copy(Q, gamma_star, x)
        # return res, 0.5 * sum(Q .* (res.^2))
        return prox_f_copy(Q, gamma_star, x), t + gamma_star #* ( 1 + tol )
    end
    return x, t
end

function epigraph_qcqp(Q, x, t)
    model = Model(Mosek.Optimizer)
    set_silent(model)

    @variable(model, p[i=1:length(x)])
    @variable(model, s)

    @objective(model, Min, (p[1] - x[1])^2 + (p[2] - x[2])^2 + (s - t)^2)

    @constraint(model, 0.5 * p' * Q * p - s <= 0)

    optimize!(model)
    return value.(model[:p]), value.(model[:s])
end

function dot_p(a, b, P)
    return LA.dot(a, P * b)
end

function norm_p(a, P)
    return sqrt(dot_p(a, a, P))
end

function primal_dual(
    x,
    v,
    L,
    sigma, 
    gamma, 
    lambda;
    MAX_ITER = 10000,
    tol = 1e-8
)
    P = [[1/gamma * LA.I(length(x)) -L']; [-L 1/sigma * LA.I(length(v))]]

    for iter = 1:MAX_ITER
        xbar = prox_f(x - gamma .* (L' * v), gamma, model.s_inds[1])
        vbar = prox_g(model, v + sigma .* (L * (2 * xbar - x)), sigma)

        rx = x - xbar
        rv = v - vbar
        rnorm = norm_p(vcat(rx, rv), P)

        if (rnorm < tol * norm_p(vcat(x, v), P))
            # Norm tolerance attained
            println("Breaking at iteration $(iter)")
            break
        end

        x = lambda * xbar + (1 - lambda) * x
        v = lambda * vbar + (1 - lambda) * v

    end

    return x
end

L = model.L
nv, nx = size(L)
x = zeros(nx); v = zeros(nv)
Lnorm = maximum(LA.svdvals(collect(L)))
gamma = (0.99) / Lnorm
sigma = gamma
lambda = 0.5

# println(primal_dual(x, v, L, sigma, gamma, lambda))
# solve_model(model, [.2, .2], tol=1e-8)

tol = 1e-12

####################################
# Prox_f is FNE (2-norm)
####################################

# Generate random numbers
for i=1:1e4  
    local x = rand(nx); local xx = copy(x)
    local y = rand(nx); local yy = copy(y)
    local gamma = rand()
    local Tx = prox_f(x, gamma, model.s_inds[1])
    local Ty = prox_f(y, gamma, model.s_inds[1])

    # Averaging property
    if !(LA.norm(Tx - Ty)^2 + LA.norm(xx - Tx - (yy - Ty))^2 <= LA.norm(xx - yy)^2 * (1. + tol))
        error("Prox_f is not FNE, iteration $(i)")
    end
end

# Reapply prox on same number
x = rand(nx); y = rand(nx); gamma = rand()
for i = 1:1e4
    local xx = copy(x); local yy = copy(y)
    local Tx = prox_f(x, gamma, model.s_inds[1])
    local Ty = prox_f(y, gamma, model.s_inds[1])

    # Averaging property
    if !(LA.norm(Tx - Ty)^2 + LA.norm(xx - Tx - (yy - Ty))^2 <= LA.norm(xx - yy)^2 * (1. + tol))
        error("Prox_f is not FNE when reapplying it, iteration $(i)")
    end
    global x = Tx
    global y = Ty
end

####################################
# Prox_g is FNE (2-norm)
####################################
for i=1:1e4  
    local x = rand(nv); local xx = copy(x)
    local y = rand(nv); local yy = copy(y)
    local gamma = rand()
    local Tx = prox_g(model, x, gamma)
    local Ty = prox_g(model, y, gamma)

    # Averaging property
    if !(LA.norm(Tx - Ty)^2 + LA.norm(xx - Tx - (yy - Ty))^2 <= LA.norm(xx - yy)^2 * (1. + tol))
        error("Prox_g is not FNE, iteration $(i)")
    end
end

# Reapply prox on same number
x = rand(nv); y = rand(nv); gamma = rand()
for i = 1:1e4
    local xx = copy(x); local yy = copy(y)
    local Tx = prox_g(model, x, gamma)
    local Ty = prox_g(model, y, gamma)

    # Averaging property
    if !(LA.norm(Tx - Ty)^2 + LA.norm(xx - Tx - (yy - Ty))^2 <= LA.norm(xx - yy)^2 * (1. + tol))
        error("Prox_g is not FNE when reapplying it, iteration $(i)")
    end
    global x = Tx
    global y = Ty
end

####################################
# CP updates are FNE (P-norm)
####################################

function pnorm(a, P)
    return sqrt(LA.dot(a, P*a))
end

# Generate random numbers
for i=1:1e4  
    local x = rand(nx); local xx = copy(x)
    local v = rand(nv); local vv = copy(v)
    local y = rand(nx); local yy = copy(y)
    local z = rand(nv); local zz = copy(z)
    local gamma = rand() / Lnorm; local sigma = rand() / Lnorm; local lambda = rand()

    local P = [[1/gamma * LA.I(nx) -L']; [-L 1/sigma * LA.I(nv)]]

    local xbar = prox_f(x - gamma .* (L' * v), gamma, model.s_inds[1])
    local vbar = prox_g(model, v + sigma .* (L * (2 * xbar - x)), sigma)

    local ybar = prox_f(y - gamma .* (L' * z), gamma, model.s_inds[1])
    local zbar = prox_g(model, z + sigma .* (L * (2 * ybar - y)), sigma)

    local Tx = lambda * xbar + (1 - lambda) * x
    local Tv = lambda * vbar + (1 - lambda) * v

    local Ty = lambda * ybar + (1 - lambda) * y
    local Tz = lambda * zbar + (1 - lambda) * z

    local T1 = vcat(Tx, Tv); local x1 = vcat(xx, vv)
    local T2 = vcat(Ty, Tz); local x2 = vcat(yy, zz)

    # Averaging property
    if !(pnorm(T1 - T2, P)^2 + pnorm(x1 - T1 - (x2 - T2), P)^2 <= pnorm(x1 - x2, P)^2 * (1. + tol))
        println(pnorm(T1 - T2, P)^2 + pnorm(x1 - T1 - (x1 - T2), P)^2)
        println(pnorm(x1 - x2, P)^2 * (1. + tol))
        error("CP update is not FNE, iteration $(i)")
    end
end

