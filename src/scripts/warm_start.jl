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

    return build_model(scen_tree, cost, dynamics, rms, DYNAMICS_IN_L_SOLVER, solver_options=SolverOptions(SuperMann)), scen_tree
end

"""
Given vectors z and v that are solutions for a problem of size N - 1, this returns intial vectors for the problem of size N
"""
function get_initial_vectors(z, d, N, nx, nu, scen_tree)
    x_inds = z_to_x(scen_tree)
    u_inds = z_to_u(scen_tree)
    s_inds = z_to_s(scen_tree)
    y_inds = z_to_y(scen_tree, 4)

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

    return z0
end

function get_sub_vector(z, i, nx, nu, scen_tree)
    ns = [i]
    childs = Int64[]
    append!(childs, scen_tree.child_mapping[i])

    while length(childs) > 0
        child = splice!(childs, 1)
        append!(ns, child)
        if child < scen_tree.leaf_node_min_index
            append!(childs, scen_tree.child_mapping[child])
        end
    end

    xnew = [z[z_to_x(scen_tree)][node_to_x(scen_tree, i)] for i in ns]
    unew = [z[z_to_u(scen_tree)][node_to_timestep(scen_tree, i)...] for i in ns if i < scen_tree.leaf_node_min_index]
    snew = [z[z_to_s(scen_tree)][node_to_s(scen_tree, i)] for i in ns]
    ynew = [z[z_to_y(scen_tree, 4)][node_to_y(scen_tree, i, 4)] for i in ns if i < scen_tree.leaf_node_min_index]
    
    xx = Float64[]; uu = Float64[]; ss = Float64[]; yy = Float64[]
    for i = 1:length(xnew)
        append!(xx, xnew[i])
    end
    for i = 1:length(unew)
        append!(uu, unew[i])
    end
    for i = 1:length(snew)
        append!(ss, snew[i])
    end
    for i = 1:length(ynew)
        append!(yy, ynew[i])
    end
    
    return vcat(
        xx,
        uu,
        ss,
        yy,
    )
end

######################
N = 4
supermann_model_2, scen_tree_2 = build_model_n(N-1, true)
supermann_model_3, scen_tree_3 = build_model_n(N, true)
d = 2; nx = 2; nu = 1
PLOT = false

###
# Solve the optimization problem
###

_, _ = solve_model(supermann_model_3, [2., 2.])
@time solve_model(supermann_model_3, [2., 2.])
z, v, x, u = solve_model(supermann_model_3, [2., 2.], return_all = true)
x0 = x[nx + 1 : 2 * nx]

z_sub = get_sub_vector(z, 2, 2, 1, scen_tree_3)
z0 = get_initial_vectors(z_sub, d, N, nx, nu, scen_tree_2)
v0 = zeros(length(v))

if PLOT
    plot_scen_tree_x(scen_tree_3, x, "output/x_original")
    plot_scen_tree_x(scen_tree_3, z0[1:length(x)], "output/x_warmup")
end

@time solve_model(supermann_model_3, x0, z0 = copy(z0), v0=copy(v0))
@time solve_model(supermann_model_3, x0, z0 = copy(z0), v0=copy(v0))
@time solve_model(supermann_model_3, x0, z0 = copy(z0), v0=copy(v0))

println("-----------")
