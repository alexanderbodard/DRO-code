module DRO

    ###
    # Problem definition
    ###
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
    
    import MathOptInterface as MOI
    import MathOptSetDistances as MOD
    import LinearAlgebra as LA

    # Random.seed!(1234)

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
    # println("x_ref: ", x_ref)
    # println("u_ref", u_ref)
    writedlm("output/log_xref.dat", x_ref, ',')

    z, v, x, u =  solve_model(vanilla_model, [2., 2.], return_all = true, tol=1e-12, verbose=false)
    z_ref = copy(z)
    v_ref = copy(v)
    @time solve_model(vanilla_model, [2., 2.], z0=ones(length(z)), v0=ones(length(v)), verbose=true)
    # @time solve_model(supermann_model, [2., 2.])
    @time solve_model(supermann_model, [2., 2.], verbose=true, tol=1e-8, z0=rand(length(z)), v0=rand(length(v)))
    # @time solve_model(supermann_model, [2., 2.], verbose=true, z0 = z * 1.001, v0 = v * 1.001)
    # x, u = solve_model(model, [2., 2.], verbose=false)
    # println("x: ", x)
    # println("u: ", u)

    writedlm("output/L.dat", vanilla_model.L, '\t')

    # error("Succes!")

    # wx = rand(length(z))
    # wv = rand(length(v))
    # gamma = (0.99) / sqrt(vanilla_model.L_norm) #rand() + 1
    # sigma = gamma#rand() / sqrt(vanilla_model.L_norm) #rand() + 1
    # lambda = rand()
    # x0 = [2., 2.]
    # for i = 1:1e3
    #     global wx
    #     global wv
    #     wxx = copy(wx)
    #     wvv = copy(wv)

    #     zxbar = wx - L_mult(vanilla_model, wv * gamma, true) - Gamma_grad_mult(vanilla_model, wx, gamma)
    #     zvbar = prox_hstar(vanilla_model, x0, wv + L_mult(vanilla_model, sigma * (2 * zxbar - wx)), sigma)

    #     if !isapprox(wv, wvv)
    #         error()
    #     end

    #     P = [[1/gamma * LA.I(length(z)) -vanilla_model.L']; [-vanilla_model.L 1/sigma * LA.I(length(v))]]
    #     P = LA.I(length(v)+length(z))
    #     r1 = LA.dot(vcat(zxbar, zvbar) - vcat(z_ref, v_ref), P * vcat(zxbar, zvbar) - vcat(z_ref, v_ref))
    #     r2 = LA.dot(vcat(wxx, wvv) - vcat(z_ref, v_ref), P * vcat(wxx, wvv) - vcat(z_ref, v_ref))
    #     if r1 >= r2
    #         println(r1, ", ", r2)
    #         error("Should be ||Tz - \bar{z}|| <= || z - \bar{z} ||, but is not in iteration $(i)")
    #     end

    #     znew = lambda * zxbar + (1 - lambda) * wx
    #     vnew = lambda * zvbar + (1 - lambda) * wv
    #     r1 = LA.dot(vcat(znew, vnew) - vcat(z_ref, v_ref), P * vcat(znew, vnew) - vcat(z_ref, v_ref))
    #     r2 = LA.dot(vcat(wxx, wvv) - vcat(z_ref, v_ref), P * vcat(wxx, wvv) - vcat(z_ref, v_ref))
    #     # || z^+ - z_ref || <= || z - z_ref ||
    #     if r1 >= r2
    #         println(r1, ", ", r2)
    #         println(lambda)
    #         # println(wv)
    #         error("Should not happen in iteration $(i)")
    #     end


    #     # # || Tz - \frac{z_{ref} + z}{2}||
    #     # r1 = sqrt(LA.dot(vcat(zxbar - (z_ref + wxx) / 2, zvbar - (v_ref + wvv) / 2), P * vcat(zxbar - (z_ref + wxx) / 2, zvbar - (v_ref + wvv) / 2)))
    #     # # || z_{ref} - z ||
    #     # r2 = sqrt(LA.dot(vcat(wxx, wvv) - vcat(z_ref, v_ref), P * (vcat(wxx, wvv) - vcat(z_ref, v_ref))))
    #     # if r1 > 0.5 * r2
    #     #     println("$(i): r1 $(r1), r2: $(r2)")
    #     #     println("Lambda $(lambda), gamma $(gamma), sigma $(sigma)")
    #     #     println("L_norm: $(vanilla_model.L_norm)")
    #     #     println(gamma * sigma * vanilla_model.L_norm < 1)
    #     #     error()
    #     # end

    #     wx = copy(znew)
    #     wv = copy(vnew)
    # end

    # println(lambda)
end