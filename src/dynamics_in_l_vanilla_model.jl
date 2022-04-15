############################################################
# Build stage
############################################################

function build_dynamics_in_l_vanilla_model(scen_tree :: ScenarioTree, cost :: Cost, dynamics :: Dynamics, rms :: Vector{Riskmeasure})
    eliminate_states = false
    n_z = get_n_z(scen_tree, rms, eliminate_states)
    z = zeros(n_z)

    n_L = get_n_L(scen_tree, rms, eliminate_states)
    L = construct_L_with_dynamics(scen_tree, rms, dynamics, n_L, n_z)
    L_trans = L'

    L_norm = maximum(LA.svdvals(collect(L)))^2
    # L_norm = sum(L.^2)

    # 4a
    inds_4a = Union{UnitRange{Int64}, Int64}[]
    offset = 0
    for k = 1:scen_tree.n_non_leaf_nodes
        n_z_part = size(rms[k].A)[2]
        ind = offset + 1 : offset + n_z_part
        append!(inds_4a, [ind])
        offset += n_z_part
    end

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
    Q_bars = []
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
        append!(Q_bars, [[Q_bars_old[i][j, j] for j = 1:size(Q_bars_old[i])[1]]])
    end

    # Compute projection
    inds_4d = Union{UnitRange{Int64}, Int64}[]
    for scen_ind  = 1:length(scenarios)
        n_z_part = length(scen_tree.min_index_per_timestep) * scen_tree.n_x + (length(scen_tree.min_index_per_timestep) - 1) * scen_tree.n_u
        ind = offset + 1 : offset + n_z_part
        append!(inds_4d, [ind])

        offset += (n_z_part + 1)
    end

    # 4e: Dynamics
    n_z_part = scen_tree.n_x * (scen_tree.n - 1)
    inds_4e = offset + 1 : offset + n_z_part

    # # Initial condition
    # z[end] = 2.

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
        zeros(n_L)
    )
end

############################################################
# Solve stage
############################################################

function primal_dual_alg(
    x, 
    v, 
    model :: DYNAMICS_IN_L_VANILLA_MODEL, 
    x0 :: Vector{Float64}; 
    DEBUG :: Bool = false, 
    tol :: Float64 = 1e-12, 
    MAX_ITER_COUNT :: Int64 = 20000,
)
    # Choose sigma and gamma such that sigma * gamma * model.L_norm < 1
    lambda = 0.5
    sigma = 0.99 / sqrt(model.L_norm)
    gamma = sigma

    # if DEBUG
    #     n_z = length(x)
    #     plot_vector = zeros(MAX_ITER_COUNT, n_z)
    #     nx = length(model.x_inds)
    #     nu = length(model.u_inds)
    #     ns = length(model.s_inds)
    #     ny = length(model.y_inds)
    #     xinit = copy(x[1:nx])
    #     residuals = zeros(MAX_ITER_COUNT)
    # end

    if DEBUG
        n_z = length(x)
        log_x = zeros(MAX_ITER_COUNT, n_z)
        nx = length(model.x_inds)
        nu = length(model.u_inds)
        ns = length(model.s_inds)
        ny = length(model.y_inds)
        xinit = copy(x[1:nx])
        log_residuals = zeros(MAX_ITER_COUNT)
    end

    # TODO: Work with some tolerance
    counter = 0
    while counter < MAX_ITER_COUNT

        # Compute xbar
        copyto!(model.v_workspace, v)
        for i = 1:length(v)
            model.v_workspace[i] *= gamma
        end
        xbar = x - L_mult(model, model.v_workspace, true) - Gamma_grad_mult(model, x, gamma)

        # Compute vbar
        for i = 1:length(x)
            model.x_workspace[i] = sigma * (2 * xbar[i] - x[i])
        end

        vbar = prox_hstar(model, x0, v + L_mult(model, model.x_workspace), sigma)

        # Compute the residual
        r_x = x - xbar
        r_v = v - vbar
        r_norm = sqrt(p_norm(r_x, r_v, r_x, r_v, model.L, gamma, sigma))

        # Update x by avering step
        x = lambda * xbar + (1 - lambda) * x

        # Update v by avering step
        for i = 1:length(v)
            v[i] *= (1 - lambda)
            v[i] += lambda * vbar[i]
        end

        if DEBUG
            log_x[counter + 1, 1:end] = x
            log_residuals[counter + 1] = r_norm
        end

        # if DEBUG
        #     plot_vector[counter + 1, 1:end] = x
        #     residuals[counter + 1] = r_norm
        # end

        if r_norm / sqrt(LA.norm(x)^2 + LA.norm(v)^2) < tol
            println("Breaking!", counter)
            break
        end
        counter += 1
    end

    # if DEBUG        
    #     residues = Float64[]
    #     for i = 1:counter
    #         append!(residues, LA.norm(plot_vector[i, 1:length(model.x_inds)] .- x_ref) / LA.norm(xinit .- x_ref))
    #     end
    #     plot(collect(1:length(residues)), log10.(residues), fmt = :png, labels=["Vanilla"])
    # end

    if DEBUG
        writedlm("output/log_vanilla_x.dat", log_x[1:counter, 1:end], ',')
        writedlm("output/log_vanilla_residual.dat", log_residuals[1:counter], ',') 
    end

    if DEBUG
        # residues = Float64[]
        # for i = 1:counter
        #     append!(residues, LA.norm(plot_vector[i, 1:length(model.x_inds)] .- x_ref) / LA.norm(xinit .- x_ref))
        # end
        # fixed_point_residues = Float64[]
        # for i = 2:counter
        #     append!(fixed_point_residues, LA.norm(plot_vector[i, 1:length(model.x_inds)] .- plot_vector[i-1, 1:length(model.x_inds)]) / LA.norm(plot_vector[i, 1:length(model.x_inds)]))
        # end

        # pgfplotsx()
        # plot(collect(1:length(fixed_point_residues)), log10.(fixed_point_residues), fmt = :png, xlims = (0, 1 * length(fixed_point_residues)), labels=["fixed_point_residue_x"])
        # filename = "fixed_point_residue_x.png"
        # savefig(filename)

        # plot!(collect(1:length(residues)), log10.(residues), fmt = :png, labels=["SuperMann"])
        # filename = "debug_x_res.png"
        # savefig(filename)

        # plot(collect(1:length(residuals)), log10.(residuals), fmt = :png, xlims = (0, 1 * length(residuals)), labels=["residual"])
        # filename = "residual.png"
        # savefig(filename)

        # plot(plot_vector[1:counter, 1 : nx], fmt = :png, labels=["x"])
        # filename = "debug_x.png"
        # savefig(filename)

        # plot(plot_vector[1:counter, nx + 1 : nx + nu], fmt = :png, labels=["u"])
        # filename = "debug_u.png"
        # savefig(filename)

        # plot(plot_vector[1:counter, nx + nu + 1 : nx + nu + ns], fmt = :png, labels=["s"])
        # filename = "debug_s.png"
        # savefig(filename)

        # plot(plot_vector[1:counter, nx + nu + ns + 1 : nx + nu + ns + ny], fmt = :png, labels=["y"])
        # filename = "debug_y.png"
        # savefig(filename)
    end

    return x
end

function solve_model(model :: DYNAMICS_IN_L_VANILLA_MODEL, x0 :: Vector{Float64}; tol :: Float64 = 1e-8, verbose :: Bool = false, return_all :: Bool = false, z0 :: Union{Vector{Float64}, Nothing} = nothing, v0 :: Union{Vector{Float64}, Nothing} = nothing)
    z = zeros(model.nz)
    v = zeros(model.nv)

    if z0 !== nothing && v0 !== nothing
        z = z0
        v = v0
    end

    z = primal_dual_alg(z, v, model, x0, tol=tol, DEBUG=verbose)

    if verbose
        println("x: ", z[model.x_inds])
        println("u: ", z[model.u_inds])
        println("s: ", z[model.s_inds])
        println("y: ", z[model.y_inds])
    end

    if return_all
        return z, v, z[model.x_inds], z[model.u_inds]
    end
    return z[model.x_inds], z[model.u_inds] 
end