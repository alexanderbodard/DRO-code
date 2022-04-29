module DRO

    ###
    # Problem definition
    ###
    using ProximalOperators, Random, JuMP, MosekTools, SparseArrays, Plots, Profile, DelimitedFiles, ForwardDiff, BenchmarkTools, CUDA

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
    N = 3; d = 2; nx = 2; nu = 1
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

    reference_model = build_model(scen_tree, cost, dynamics, rms, MOSEK_SOLVER)
    vanilla_model = build_model(scen_tree, cost, dynamics, rms, DYNAMICS_IN_L_SOLVER)
    supermann_model = build_model(scen_tree, cost, dynamics, rms, DYNAMICS_IN_L_SOLVER, solver_options=SolverOptions(true))

    ###
    # Solve the optimization problem
    ###

    # @time solve_model(reference_model, [2., 2.])
    x_ref, u_ref, s_ref, y_ref = solve_model(reference_model, [2., 2.])
    println(x_ref)

    # z, v, x, u =  solve_model(vanilla_model, [2., 2.], return_all = true, tol=1e-12, verbose=false)
    # @time solve_model(vanilla_model, [2., 2.], verbose=false)
    @time solve_model(vanilla_model, [2., 2.], verbose=false, z0=zeros(vanilla_model.nz), v0=zeros(vanilla_model.nv))
    @time solve_model(vanilla_model, [2., 2.], verbose=false, z0=zeros(vanilla_model.nz), v0=zeros(vanilla_model.nv))
    # @time solve_model(supermann_model, [2., 2.], verbose=false, z0=zeros(vanilla_model.nz), v0=zeros(vanilla_model.nv))
    # @time solve_model(supermann_model, [2., 2.], verbose=false, z0=zeros(vanilla_model.nz), v0=zeros(vanilla_model.nv))
    # println("x: ", x)
    # println("u: ", u)

    println(vanilla_model.z)
    # println(vanilla_model.Q_bars)
    # pgfplotsx()
    # spy(vanilla_model.L)
    # savefig("saved_output/spy.png")

    # writedlm("output/L.dat", vanilla_model.L, '\t')
    # writedlm("output/log_xref.dat", vcat(x_ref, u_ref, s_ref, y_ref), ',')
end
