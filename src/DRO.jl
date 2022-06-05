module DRO

    ###
    # Problem definition
    ###
    using ProximalOperators, Random, JuMP, MosekTools, SparseArrays, Plots, Profile, DelimitedFiles, ForwardDiff, BenchmarkTools#, CUDA

    if (@isdefined GPU)
      using CUDA
    end

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

    include("builtin_models.jl")
    
    import MathOptInterface as MOI
    import MathOptSetDistances as MOD
    import LinearAlgebra as LA

    Random.seed!(1234)

    export get_tp1, get_tp2

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
    T_s = 0.1
    A = [[[1.,0.] [T_s, 1.0 - (i - 1) / d * T_s]] for i in 1:d]
    B = [reshape([0., T_s], :, 1) for _ in 1:d]
    dynamics = get_uniform_dynamics(A, B)

    # Cost: Let's take a quadratic cost, equal at all timesteps
    Q = LA.Matrix([2.2 0; 0 3.7])
    R = reshape([3.2], 1, 1)
    cost = get_uniform_cost(Q, R, N)

    # Risk measures: Risk neutral: A = I, B = [I; -I], b = [1;1;-1;-1]
    """
    Risk neutral: A = I, B = [I; -I], b = [0.5;0.5;-0.5;-0.5]
    AVaR: A = I, B = [-I, I, 1^T, -1^T], b = [0; p / alpha; 1, -1]
    """
    rms = get_uniform_rms_robust(d, N)

    p_ref = [0.5, 0.5]; alpha=0.01
    rms = get_uniform_rms_avar(p_ref, alpha, d, N)
    # rms = get_uniform_rms_tv(p_ref, 0., d, N)
    # rms = get_uniform_rms_risk_neutral(p_ref, d, N)

    ###
    # Formulate the optimization problem
    ###

    # reference_model = build_model(scen_tree, cost, dynamics, rms, MOSEK_SOLVER)
    vanilla_model = build_model(scen_tree, cost, dynamics, rms, DYNAMICS_IN_L_SOLVER)
    # println(vanilla_model.L_norm)
    # supermann_model = build_model(scen_tree, cost, dynamics, rms, DYNAMICS_IN_L_SOLVER, solver_options=SolverOptions(true))

    ###
    # Solve the optimization problem
    ###

    # @time solve_model(reference_model, [2., 2.])
    # x_ref, u_ref, s_ref, y_ref = solve_model(reference_model, [2., 2.])
    # println(x_ref[1:4])

    # z, v, x, u =  solve_model(vanilla_model, [2., 2.], return_all = true, tol=1e-12, verbose=false)
    # @time solve_model(vanilla_model, [2., 2.], verbose=false, z0=zeros(vanilla_model.nz), v0=zeros(vanilla_model.nv))
    # @time solve_model(vanilla_model, [2., 2.], verbose=PRINT_CL, z0=zeros(vanilla_model.nz), v0=zeros(vanilla_model.nv), MAX_ITER_COUNT=1000000)
    # @time solve_model(vanilla_model, [2., 2.], verbose=PRINT_CL, z0=zeros(vanilla_model.nz), v0=zeros(vanilla_model.nv), MAX_ITER_COUNT=1000000)
    @time solve_model(vanilla_model, [2., 2.], verbose=PRINT_CL, z0=zeros(vanilla_model.nz), v0=zeros(vanilla_model.nv), MAX_ITER_COUNT=1000000, tol=1e-6)
    # @time solve_model(vanilla_model, [3.01, 0.83], verbose=false)
    # @time solve_model(supermann_model, [2., 2.], verbose=false, z0=zeros(supermann_model.nz), v0=zeros(supermann_model.nv))
    # @time solve_model(supermann_model, [2., 2.], verbose=false, z0=zeros(vanilla_model.nz), v0=zeros(vanilla_model.nv))
    # println("x: ", x)
    # println("u: ", u)

    pgfplotsx()

    # println(vanilla_model.v[vanilla_model.inds_4d])
    res = Float64[]
    for i in vanilla_model.inds_4d
      if abs(vanilla_model.v[i]) > 1e-6
        append!(res, abs(vanilla_model.v[i]))
      end
    end
    # p = scatter(abs.(vanilla_model.v[vanilla_model.inds_4d]), zeros(length(vanilla_model.inds_4d)), xaxis=:log, ylims=(-1., 1.))
    # p = scatter(res, zeros(length(res)), ylims=(-1, 1))
    # display(p)

    p = histogram(abs.(vanilla_model.v[vanilla_model.inds_4d]), bins=10.0 .^ (-20:2), xaxis=:log, xlims=(1e-20, 100))
    display(p)

    # println(vanilla_model.z[1:4])
    # println(vanilla_model.Q_bars)
    # pgfplotsx()
    # spy(vanilla_model.L)
    # savefig("saved_output/spy.png")

    # writedlm("output/L.dat", vanilla_model.L, '\t')
    # writedlm("output/log_xref.dat", vcat(x_ref, u_ref, s_ref, y_ref), ',')

    # model, ref_model = get_tm1()

    # ### Determine reference solution
    # x_ref, u_ref, s_ref, y_ref = solve_model(ref_model, [2., 2.])
    # writedlm("output/reference_solution.dat", x_ref, ',')

    # ### Run vanilla solver
    # solve_model(
    #   model, 
    #   [2., 2.], 
    #   verbose=DRO.PRINT_AND_WRITE, 
    #   filename = "output/residuals.dat", 
    #   z0=zeros(model.nz), 
    #   v0=zeros(model.nv),
    #   tol=1e-4
    # )

  # # Scenario tree
  # N = 5; d = 2; nx = 2; nu = 1
  # scen_tree = generate_scenario_tree(N, d, nx, nu)

  # # Dynamics: Based on a discretized car model
  # T_s = 0.1
  # A = [[[1.,0.] [T_s, 1.0 - rand()*T_s]] for _ in 1:d]
  # B = [reshape([0., T_s], :, 1) for _ in 1:d]
  # dynamics = get_uniform_dynamics(A, B)

  # # Cost: Let's take a quadratic cost, equal at all timesteps
  # Q = LA.Matrix([2.2 0; 0 3.7])
  # R = reshape([3.2], 1, 1)
  # cost = get_uniform_cost(Q, R, N)

  # # Risk measures: Risk neutral: A = I, B = [I; -I], b = [1;1;-1;-1]
  # """
  # Risk neutral: A = I, B = [I; -I], b = [0.5;0.5;-0.5;-0.5]
  # AVaR: A = I, B = [-I, I, 1^T, -1^T], b = [0; p / alpha; 1, -1]
  # """
  # rms = get_uniform_rms_robust(d, N)

  # p_ref = [0.5, 0.5]; alpha=.1
  # rms = get_uniform_rms_avar(p_ref, alpha, d, N)

  # ###
  # # Formulate the optimization problem
  # ###

  # vanilla_model_2 = build_model(scen_tree, cost, dynamics, rms, DYNAMICS_IN_L_SOLVER)

  # z_warm = zeros(vanilla_model_2.nz)
  # z_warm[1:14] = vanilla_model.z[1:14] # x overnemen

  # # x extrapoleren op N = 4
  # z_warm[15:16] = vanilla_model.z[7:8]
  # z_warm[17:18] = vanilla_model.z[7:8]

  # z_warm[19:20] = vanilla_model.z[9:10]
  # z_warm[21:22] = vanilla_model.z[9:10]

  # z_warm[23:24] = vanilla_model.z[11:12]
  # z_warm[25:26] = vanilla_model.z[11:12]

  # z_warm[27:28] = vanilla_model.z[13:14]
  # z_warm[29:30] = vanilla_model.z[13:14]

  # # x extrapoleren op N = 5
  # z_warm[31:32] = vanilla_model.z[7:8]
  # z_warm[33:34] = vanilla_model.z[7:8]
  # z_warm[35:36] = vanilla_model.z[7:8]
  # z_warm[37:38] = vanilla_model.z[7:8]

  # z_warm[39:40] = vanilla_model.z[9:10]
  # z_warm[41:42] = vanilla_model.z[9:10]
  # z_warm[43:44] = vanilla_model.z[9:10]
  # z_warm[45:46] = vanilla_model.z[9:10]

  # z_warm[47:48] = vanilla_model.z[11:12]
  # z_warm[49:50] = vanilla_model.z[11:12]
  # z_warm[51:52] = vanilla_model.z[11:12]
  # z_warm[53:54] = vanilla_model.z[11:12]

  # z_warm[55:56] = vanilla_model.z[13:14]
  # z_warm[57:58] = vanilla_model.z[13:14]
  # z_warm[59:60] = vanilla_model.z[13:14]
  # z_warm[61:62] = vanilla_model.z[13:14]

  # @time solve_model(vanilla_model_2, [2., 2.], verbose=PRINT_CL, z0=zeros(vanilla_model_2.nz), v0=zeros(vanilla_model_2.nv), MAX_ITER_COUNT=1000000)
  # @time solve_model(vanilla_model_2, [2., 2.], verbose=PRINT_CL, z0=zeros(vanilla_model_2.nz), v0=zeros(vanilla_model_2.nv), MAX_ITER_COUNT=1000000)
  # @time solve_model(vanilla_model_2, [2., 2.], verbose=PRINT_CL, z0=z_warm, v0=zeros(vanilla_model_2.nv), MAX_ITER_COUNT=1000000)

  # @time solve_model(vanilla_model_2, [2., 2.], verbose=PRINT_CL, z0=(vanilla_model_2.z .+ (0.3 * rand(vanilla_model_2.nz))), v0=(vanilla_model_2.v .+ (0.3 * rand(vanilla_model_2.nv))), MAX_ITER_COUNT=1000000)
  # @time solve_model(vanilla_model_2, [2., 2.], verbose=PRINT_CL, z0=(vanilla_model_2.z .+ (0.1 * rand(vanilla_model_2.nz))), v0=zeros(vanilla_model_2.nv), MAX_ITER_COUNT=1000000)
end
