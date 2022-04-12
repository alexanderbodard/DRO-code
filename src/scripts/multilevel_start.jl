using ProximalOperators, Random, JuMP, MosekTools, SparseArrays, Plots, Profile

include("../scenario_tree.jl")
include("../risk_constraints.jl")
include("../dynamics.jl")
include("../cost.jl")

include("../model.jl")
include("../custom_model.jl")
include("../dynamics_in_l_vanilla_model.jl")
include("../dynamics_in_l_supermann_model.jl")
include("../mosek_model.jl")

import MathOptInterface as MOI
import MathOptSetDistances as MOD
import LinearAlgebra as LA

Random.seed!(1234)

##########################
##########################

function build_model_n(N :: Int64, SuperMann :: Bool)
    # Scenario tree
    d = 2; nx = 2; nu = 1
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

    ###
    # Formulate the optimization problem
    ###

    return build_model(scen_tree, cost, dynamics, rms, DYNAMICS_IN_L_SOLVER, solver_options=SolverOptions(SuperMann))
end

"""
Given vectors z and v that are solutions for a problem of size N - 1, this returns intial vectors for the problem of size N
"""
function get_initial_vectors(z, v, model, d, N, nx, nu)
    x_inds = model.x_inds
    u_inds = model.u_inds
    s_inds = model.s_inds
    y_inds = model.y_inds

    n_leafs = d^(N - 1)
    n_leaf_parents = Int(n_leafs / d)

    # x
    x_offset = Int((d^(N - 2) - 1) / (d - 1)) * nx
    x_new = [z[x_offset + (((i - 1) ÷ d)) * nx + 1 : x_offset + ((i - 1) ÷ d + 1) * nx] for i = 1:n_leafs]

    # s
    s_offset = s_inds[1] - 1 + Int((d^(N - 2) - 1) / (d - 1)) * 1
    s_new = [z[s_offset + (((i - 1) ÷ d)) * 1 + 1 : s_offset + ((i - 1) ÷ d + 1) * 1] for i = 1:n_leafs]

    # y
    y_offset = y_inds[1] - 1 + Int((d^(N - 3) - 1) / (d - 1)) * 4
    y_new = [z[y_offset + (((i - 1) ÷ d)) * 4 + 1 : y_offset + ((i - 1) ÷ d + 1) * 4] for i = 1:n_leaf_parents]

    z0 = vcat(
        z[x_inds], x_new..., 
        z[u_inds], zeros(nu), 
        z[s_inds], s_new..., 
        z[y_inds], y_new...
    )
    v0 = zeros(supermann_model_3.nv)

    return z0, v0
end

######################
N = 3

supermann_model_2 = build_model_n(N - 1, true)
supermann_model_3 = build_model_n(N, true)

###
# Solve the optimization problem
###

_, _ = solve_model(supermann_model_3, [2., 2.])
@time solve_model(supermann_model_3, [2., 2.])

z, v, x, u = solve_model(supermann_model_2, [2., 2.], return_all = true)
@time solve_model(supermann_model_2, [2., 2.])

d = 2; nx = 2; nu = 1
z0, v0 = get_initial_vectors(z, v, supermann_model_2, d, N, nx, nu)

@time solve_model(supermann_model_3, [2., 2.], z0 = copy(z0), v0=copy(v0))
@time solve_model(supermann_model_3, [2., 2.], z0 = copy(z0), v0=copy(v0))
@time solve_model(supermann_model_3, [2., 2.], z0 = copy(z0), v0=copy(v0))

println("-----------")
