if @isdefined GPU
  include("cuda/dynamics_in_l_vanilla_model.jl")
else
  include("sequential/dynamics_in_l_vanilla_model.jl")
end

############################################################
# Build stage
############################################################

function build_dynamics_in_l_vanilla_model(scen_tree :: ScenarioTree, cost :: Cost, dynamics :: Dynamics, rms :: Vector{Riskmeasure}; gpu :: Bool = false)
    eliminate_states = false
    n_z = get_n_z(scen_tree, rms, eliminate_states)
    z = zeros(n_z)

    n_L = get_n_L(scen_tree, rms, eliminate_states)
    L = construct_L_with_dynamics(scen_tree, rms, dynamics, n_L, n_z)
    # L_trans = L'

    L_norm = maximum(LA.svdvals(collect(L)))^2
    # L_norm = sum(L.^2)
    # L_norm = 2.0^(N-9) * 259

    # 4a
    inds_4a_start = 1
    offset = 0
    for k = 1:scen_tree.n_non_leaf_nodes
        n_z_part = size(rms[k].A)[2]
        # ind = offset + 1 : offset + n_z_part
        # append!(inds_4a, [ind])
        offset += n_z_part
    end
    inds_4a_end = offset
    inds_4a = inds_4a_start : inds_4a_end

    # 4b
    inds_4b_start = offset + 1
    for k = 1:scen_tree.n_non_leaf_nodes
        n_z_part = length(rms[k].b)
        offset += n_z_part
    end
    inds_4b_end = offset
    inds_4b = inds_4b_start : inds_4b_end

    # 4c
    inds_4c = Union{UnitRange{Int64}, Int64}[]
    b_bars = Vector{Float64}[]
    for k = 1:scen_tree.n_non_leaf_nodes
        n_z_part = length(rms[k].b) + 1
        ind = offset + 1 : offset + n_z_part
        b_bar = [1; rms[k].b]
        append!(inds_4c, [ind])
        append!(b_bars, [b_bar])

        offset += n_z_part
    end

    # 4d: Cost epigraph projection
    # TODO: Extract into separate scenario tree method
    scenarios = []
    for k = scen_tree.leaf_node_min_index:scen_tree.leaf_node_max_index
        nn = k
        scenario = [nn]
        while nn != 1
            nn = scen_tree.anc_mapping[nn]
            append!(scenario, nn)
        end
        append!(scenarios, [reverse(scenario)])
    end
    ####
    R_offset = length(scen_tree.min_index_per_timestep) * scen_tree.n_x
    T = length(scen_tree.min_index_per_timestep)
    Q_bars = Float64[]
    Q_bars_old = []
    # Q_bar is a block diagonal matrix with the corresponding Q's and R's for that scenario
    for scen_ind = 1:length(scenarios)
        scenario = scenarios[scen_ind]

        L_I = Float64[]
        L_J = Float64[]
        L_V = Float64[]

        # TODO: This computation can be simplified a LOT
        for t = 1:T
            nn = scenario[t]
            # Add Q to Q_bar
            Q_I, Q_J, Q_V = findnz(sparse(cost.Q[t]))

            append!(L_I, Q_I .+ (t - 1) * scen_tree.n_x)
            append!(L_J, Q_J .+ (t - 1) * scen_tree.n_x)
            append!(L_V, 2 .* Q_V)

            if t < T
                # Add R to Q_bar
                R_I, R_J, R_V = findnz(sparse(cost.R[t]))

                append!(L_I, R_I .+ R_offset .+ (t - 1) * scen_tree.n_u)
                append!(L_J, R_J .+ R_offset .+ (t - 1) * scen_tree.n_u)
                append!(L_V, 2 .* R_V)
            end
        end

        append!(Q_bars_old, [sparse(L_I, L_J, L_V, 
            length(scen_tree.min_index_per_timestep) * scen_tree.n_x + (length(scen_tree.min_index_per_timestep) - 1) * scen_tree.n_u, 
            length(scen_tree.min_index_per_timestep) * scen_tree.n_x + (length(scen_tree.min_index_per_timestep) - 1) * scen_tree.n_u
        )])
    end
    for i = 1:length(Q_bars_old)
        append!(Q_bars, [Q_bars_old[i][j, j] for j = 1:size(Q_bars_old[i])[1]])
    end

    # Compute projection
    inds_4d = Int64[]
    for scen_ind  = 1:length(scenarios)
        n_z_part = length(scen_tree.min_index_per_timestep) * scen_tree.n_x + (length(scen_tree.min_index_per_timestep) - 1) * scen_tree.n_u
        ind = offset + 1 : offset + n_z_part
        append!(inds_4d, collect(ind))

        offset += (n_z_part + 1)
    end

    # 4e: Dynamics
    n_z_part = scen_tree.n_x * (scen_tree.n - 1)
    inds_4e = offset + 1 : offset + n_z_part

    # # Initial condition
    # z[end] = 2.

    if gpu
      return DYNAMICS_IN_L_VANILLA_MODEL(
          L,
          L_norm,
          n_z,
          n_L,
          z_to_x(scen_tree),
          z_to_u(scen_tree),
          z_to_s(scen_tree),
          z_to_y(scen_tree, 4),
          inds_4a,
          inds_4b,
          inds_4c,
          b_bars,
          CuArray(inds_4d),
          CuArray(Q_bars),
          inds_4e,
          zeros(
              length(scen_tree.min_index_per_timestep) * scen_tree.n_x + 
              (length(scen_tree.min_index_per_timestep) - 1) * scen_tree.n_u
          ),
          zeros(n_z),
          zeros(n_L),
          Mem.pin(Array{Float64}(undef, n_L)), #zeros(n_L),
          CuArray(zeros(n_L)),
          CuArray(zeros(n_L)),
          zeros(n_z),
          zeros(n_L),
          zeros(n_z),
          zeros(n_L),
          zeros(n_z),
          zeros(n_L),
          zeros(scen_tree.n_x),
          length(scen_tree.min_index_per_timestep) * scen_tree.n_x + (length(scen_tree.min_index_per_timestep) - 1) * scen_tree.n_u
      )
    else
      return DYNAMICS_IN_L_VANILLA_MODEL(
        L,
        L_norm,
        n_z,
        n_L,
        z_to_x(scen_tree),
        z_to_u(scen_tree),
        z_to_s(scen_tree),
        z_to_y(scen_tree, 4),
        inds_4a,
        inds_4b,
        inds_4c,
        b_bars,
        inds_4d,
        Q_bars,
        inds_4e,
        zeros(
            length(scen_tree.min_index_per_timestep) * scen_tree.n_x + 
            (length(scen_tree.min_index_per_timestep) - 1) * scen_tree.n_u
        ),
        zeros(n_z),
        zeros(n_L),
        zeros(n_L),
        zeros(n_L),
        zeros(n_L),
        zeros(n_z),
        zeros(n_L),
        zeros(n_z),
        zeros(n_L),
        zeros(n_z),
        zeros(n_L),
        zeros(scen_tree.n_x),
        length(scen_tree.min_index_per_timestep) * scen_tree.n_x + (length(scen_tree.min_index_per_timestep) - 1) * scen_tree.n_u
    )
  end
end

############################################################
# Solve stage
############################################################

function prox_f!(
    model :: DYNAMICS_IN_L_VANILLA_MODEL,
    arg :: Vector{Float64},
    gamma :: Float64
)
    copyto!(model.zbar, arg)
    model.zbar[model.s_inds[1]] -= gamma    
end

function update_zvbar!(
    model :: DYNAMICS_IN_L_VANILLA_MODEL,
    gamma :: Float64,
    sigma :: Float64
)
    ### Compute zbar
    copyto!(model.z_workspace, model.z)
    # z_workspace = -gamma * L' * v + z_workspace
    LA.mul!(model.z_workspace, model.L', model.v, -gamma, 1.)

    # model.z_workspace[1:end] = model.L' * model.v
    # @simd for i = 1:model.nz
    #     @inbounds @fastmath model.z_workspace[i] = -gamma * model.z_workspace[i] + model.z[i]
    # end

    prox_f!(model, model.z_workspace , gamma)

    ### Compute vbar
    copyto!(model.z_workspace, model.z)
    # z_workspace = 2 * zbar - z_workspace
    LA.BLAS.axpby!(2., model.zbar, -1., model.z_workspace)

    copyto!(model.v_workspace, model.v)
    # v_workspace = sigma * L * z_workspace + v_workspace
    LA.mul!(model.v_workspace, model.L, model.z_workspace, sigma, 1.)

    # model.v_workspace[1:end] = model.L * model.z_workspace
    # @simd for i = 1:model.nv
    #     @inbounds @fastmath model.v_workspace[i] = sigma * model.v_workspace[i] + model.v[i]
    # end

    prox_g!(model, model.v_workspace, sigma)
end

function get_rnorm(
    model :: DYNAMICS_IN_L_VANILLA_MODEL,
    gamma :: Float64,
    sigma :: Float64
)
    # TODO: Implement this properly
    return sqrt(LA.dot(model.rz, model.rz) + LA.dot(model.rv, model.rv))

    return sqrt(
        LA.dot(model.rz, model.rz) / gamma 
        + LA.dot(model.rv, model.rv) / sigma
        - LA.dot(model.L * model.rz, model.rv) 
        - LA.dot(model.rv, model.L * model.rz)
    )
end

function update_residual!(
    model :: DYNAMICS_IN_L_VANILLA_MODEL,
    gamma :: Float64,
    sigma :: Float64
)
    ### Update model.rz
    copyto!(model.rz, model.z)
    # rz = z - zbar
    LA.BLAS.axpy!(-1., model.zbar, model.rz)

    ### Update model.rv
    copyto!(model.rv, model.v)
    # rv = v - vbar
    LA.BLAS.axpy!(-1., model.vbar, model.rv)

    ### Update model.rnorm
    return get_rnorm(model, gamma, sigma)
end

function update_zv!(
    model :: DYNAMICS_IN_L_VANILLA_MODEL,
    lambda :: Float64
)
    ### Update z
    # z = lambda * zbar + (1 - lambda) * z
    # LA.BLAS.axpby!(lambda, model.zbar, 1 - lambda, model.z)
    @simd for i = 1:model.nz
        @inbounds @fastmath model.z[i] = lambda * model.zbar[i] + (1 - lambda) * model.z[i]
    end

    ### Update v
    # v = lambda * vbar + (1 - lambda) * v
    # LA.BLAS.axpby!(lambda, model.vbar, 1 - lambda, model.v)
    @simd for i = 1:model.nv
        @inbounds @fastmath model.v[i] = lambda * model.vbar[i] + (1 - lambda) * model.v[i]
    end
end

function primal_dual_alg!(
    model :: DYNAMICS_IN_L_VANILLA_MODEL;
    MAX_ITER_COUNT :: Int64 = 100000,
    tol :: Float64 = 1e-8,
    verbose :: VERBOSE_LEVEL = SILENT,
    path = "logs/",
    filename = "logs",
    log_stride :: Int64 = 1,
    gamma :: Union{Float64, Nothing} = nothing,
    sigma :: Union{Float64, Nothing} = nothing
)
    iter = 0
    rnorm = Inf
    rnorm_0 = 0.
    
    # Choose sigma and gamma such that sigma * gamma * model.L_norm < 1
    lambda = 0.5
    if sigma === nothing
      sigma = 0.99 / sqrt(model.L_norm)
    end
    if gamma === nothing
      gamma = sigma
    end

    # Preallocate extra memory for logging 
    if verbose == PRINT_AND_WRITE
      println("Starting solve step...")

      nx = length(model.x_inds)
      n_iter_log = Int(floor(MAX_ITER_COUNT / log_stride))
      rnorms = zeros(n_iter_log)
      xs = zeros(n_iter_log, nx)
    end

    while iter < MAX_ITER_COUNT
      if iter === 1
      	rnorm_0 = rnorm
      end

        update_zvbar!(model, gamma, sigma)
        rnorm = update_residual!(model, gamma, sigma)
        update_zv!(model, lambda)

        if verbose == PRINT_AND_WRITE && iter % log_stride == 0
          rnorms[iter ÷ log_stride + 1] = rnorm
          xs[(iter ÷ log_stride +1), :] = model.z[model.x_inds]
        end

        if rnorm < tol * rnorm_0
            if verbose == PRINT_CL || verbose == PRINT_AND_WRITE
              println("Breaking!", iter)
            end
            break
        end

        iter += 1
    end

    # Write away logs
    if verbose == PRINT_AND_WRITE
      println("Writing logs to output file...")

      writedlm(path * filename * "_residual.dat", rnorms[1:iter ÷ log_stride], ',')
      writedlm(path * filename * "_x.dat", xs[1:iter ÷ log_stride, :], ',')
      println("Finished logging.")
    end
end

function solve_model(
  model :: DYNAMICS_IN_L_VANILLA_MODEL, 
  x0 :: Vector{Float64}; 
  tol :: Float64 = 1e-8, 
  MAX_ITER_COUNT :: Int64 = 100000,
  verbose :: VERBOSE_LEVEL = SILENT,
  path = "logs/",
  filename  = "logs",
  log_stride :: Int64 = 1,
  return_all :: Bool = false, 
  z0 :: Union{Vector{Float64}, Nothing} = nothing, 
  v0 :: Union{Vector{Float64}, Nothing} = nothing,
  gamma :: Union{Float64, Nothing} = nothing,
  sigma :: Union{Float64, Nothing} = nothing
)
    if z0 !== nothing && v0 !== nothing
        copyto!(model.z, z0)
        copyto!(model.v, v0)
    end

    copyto!(model.x0, x0)

    primal_dual_alg!(model, tol=tol, verbose=verbose, filename = filename, path=path, MAX_ITER_COUNT = MAX_ITER_COUNT, log_stride = log_stride, sigma = sigma, gamma = gamma)
end
