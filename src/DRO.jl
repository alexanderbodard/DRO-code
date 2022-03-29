module DRO

    include("cost.jl")
    include("dynamics.jl")
    include("model.jl")
    include("risk_constraints.jl")
    include("scenario_tree.jl")

    using ProximalOperators, Random, JuMP, MosekTools, SparseArrays, Plots
    import MathOptInterface as MOI
    import MathOptSetDistances as MOD
    import LinearAlgebra as LA
    Random.seed!(1234)

    ##########################
    # Mosek reference implementation
    ##########################

    ###
    # Problem statement
    ###

    # Scenario tree
    node_info = [
        ScenarioTreeNodeInfo(
            collect((i - 1) * 2 + 1 : i * 2),
            i < 4 ? [i] : nothing,
            i > 1 ? (i % 2) + 1 : nothing,
            i,
        ) for i in collect(1:7)
    ]

    scen_tree = ScenarioTree(
        Dict(
            1 => [2, 3], 
            2 => [4, 5], 
            3 => [6, 7],
        ),
        Dict(
            2 => 1,
            3 => 1,
            4 => 2,
            5 => 2,
            6 => 3,
            7 => 3,
        ),
        node_info,
        2,
        1,
        7,
        3,
        4,
        7,
        [1, 2, 4]
    )

    # Dynamics: Based on a discretized car model
    T_s = 0.05
    n_x = 2
    n_u = 1
    d = 2
    A = [[[1.,0.] [T_s, 1.0 - rand()*T_s]] for _ in 1:d]
    B = [reshape([0., T_s], :, 1) for _ in 1:d]
    dynamics = Dynamics(A, B, n_x, n_u)

    # Cost: Let's take a quadratic cost, equal at all timesteps
    Q = LA.Matrix([2.2 0; 0 3.7])
    R = reshape([3.2], 1, 1)
    cost = Cost(
        collect([
            Q for _ in 1:3
        ]),
        collect([
            R for _ in 1:3
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

    model = build_model(scen_tree, CUSTOM_SOLVER)

    ###
    # Solve the optimization problem
    ###

    x, u = solve_model(model, CUSTOM_SOLVER)

    # plot_scen_tree_x(scen_tree, x, "x")
    # plot_scen_tree_x_i(scen_tree, x, 1, "x_1")
    # plot_scen_tree_x_i(scen_tree, x, 2, "x_2")
end