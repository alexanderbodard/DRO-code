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
    N = 2; d = 2; nx = 1; nu = 1
    scen_tree = generate_scenario_tree(N, d, nx, nu)

    # Dynamics: Based on a discretized car model
    T_s = 0.05
    # n_x = 2
    n_x = 1
    n_u = 1
    d = 2
    # A = [[[1.,0.] [T_s, 1.0 - rand()*T_s]] for _ in 1:d]
    # B = [reshape([0., T_s], :, 1) for _ in 1:d]
    A = [reshape([i / 4], 1, 1) for i = 1:d]
    B = [reshape([T_s], :, 1) for _ in 1:d]
    dynamics = Dynamics(A, B, n_x, n_u)

    # Cost: Let's take a quadratic cost, equal at all timesteps
    # Q = LA.Matrix([2.2 0; 0 3.7])
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

    reference_model = build_model(scen_tree, MOSEK_SOLVER)
    model = build_model(scen_tree, H_X_SOLVER)

    ###
    # Solve the optimization problem
    ###

    x_ref, u_ref, s_ref, y_ref = solve_model(reference_model, MOSEK_SOLVER)
    println(x_ref)
    println(u_ref)
    println(s_ref)
    println(y_ref)

    x, u = solve_model(model, H_X_SOLVER)
    println("x: ", x)
    println("u: ", u)
    # @time solve_model(model, H_X_SOLVER)

    L_II, L_JJ, L_VV = construct_L_4e(scen_tree, dynamics, length(x) + length(u))
    H = sparse(L_II, L_JJ, L_VV, scen_tree.n_x * (scen_tree.n - 1), scen_tree.n * scen_tree.n_x + (scen_tree.n_non_leaf_nodes) * scen_tree.n_u)

    # println("--")
    # # println(dynamics.A[1] * x[1:2] + dynamics.B[1] * u[1])
    # println(vcat(x_ref, u_ref))
    # println(vcat(x, u))
    println(H * vcat(x, u))
    # println(H * vcat(x_ref, u_ref))
    # display(collect(H)[1:6, 1:end])
    # display(collect(H)[7:12, 1:end])

    # plot_scen_tree_x(scen_tree, x, "x")
    # plot_scen_tree_x_i(scen_tree, x, 1, "x_1")
    # plot_scen_tree_x_i(scen_tree, x, 2, "x_2")
end