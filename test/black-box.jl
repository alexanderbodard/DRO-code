using ProximalOperators, Random, JuMP, MosekTools, SparseArrays, Plots

include("../src/scenario_tree.jl")
include("../src/dynamics.jl")
include("../src/cost.jl")
include("../src/risk_constraints.jl")

include("../src/model.jl")
include("../src/custom_model.jl")

include("../src/dynamics_in_l_vanilla_model.jl")
include("../src/mosek_model.jl")

import MathOptInterface as MOI
import MathOptSetDistances as MOD
import LinearAlgebra as LA
Random.seed!(1234)

@testset "Small model" begin

    ###
    # Problem statement
    ###

    # Scenario tree
    N = 2; d = 2; nx = 1; nu = 1
    scen_tree = generate_scenario_tree(N, d, nx, nu)

    # Dynamics: Based on a discretized car model
    T_s = 0.05
    A = [reshape([i / 4], 1, 1) for i = 1:d]
    B = [reshape([T_s], :, 1) for _ in 1:d]
    dynamics = Dynamics(A, B, nx, nu)

    # Cost: Let's take a quadratic cost, equal at all timesteps
    Q = reshape([2.2], 1, 1)
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

    reference_model = build_model(scen_tree, cost, dynamics, rms, MOSEK_SOLVER)
    model = build_model(scen_tree, cost, dynamics, rms, DYNAMICS_IN_L_SOLVER)

    ###
    # Solve the optimization problem
    ###

    x_ref, u_ref, s_ref, y_ref = solve_model(reference_model, [2.])
    x, u = solve_model(model, [2.], tol=1e-12)

    @test isapprox(x, x_ref, rtol = 1e-5)
    @test isapprox(u, u_ref, rtol = 1e-5)
end