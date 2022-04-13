############################################################
# Build stage
############################################################

function build_dynamics_in_l_supermann_model(scen_tree :: ScenarioTree, cost :: Cost, dynamics :: Dynamics, rms :: Vector{Riskmeasure})
    eliminate_states = false
    n_z = get_n_z(scen_tree, rms, eliminate_states)
    z = zeros(n_z)

    n_L = get_n_L(scen_tree, rms, eliminate_states)
    L = construct_L_with_dynamics(scen_tree, rms, dynamics, n_L, n_z)
    L_trans = L'

    L_norm = maximum(LA.svdvals(collect(L)))^2

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

    return DYNAMICS_IN_L_SUPERMANN_MODEL(
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

function broyden_sherman_morrison(H, delta_z, delta_R, L, alpha1, alpha2; theta_bar = 0.5)
    gamma = LA.dot(H * delta_R, delta_z) / LA.norm(delta_z)^2
    if abs(gamma) >= theta_bar
        theta = 1
    elseif gamma === 0 # In Julia, sign(0) = 0, whereas we define it as sign(0) = 1; handle separately
        theta = (1 - theta_bar)
    else
        theta = (1 - sign(gamma) * theta_bar) / (1 - gamma)
    end

    s_tilde = (1 - theta) * delta_z + theta * H * delta_R # With powell

    return H + 1 / (delta_z' * (s_tilde)) * (delta_z - (s_tilde)) * (delta_z' * H)
end

function P_mult(x, L, alpha1, alpha2)
    x1 = x[1:size(L)[2]]; x2 = x[size(L)[2] + 1 : end]

    return vcat(x1 / alpha1 - L' * x2, -L * x1 + x2 / alpha2)
end

function broyden(Sbuf, Stildebuf, PSbuf, d, s, y, rx, k, L, alpha1, alpha2; MAX_K = 20, theta_bar = 0.)
    Ps = P_mult(s, L, alpha1, alpha2)
    d = -rx
    stilde = y
    n = length(s)
    for i = 1 : k
        inds = (i - 1) * n + 1 : i * n
        stilde += LA.dot(Sbuf[inds], stilde) / LA.dot(Sbuf[inds], Stildebuf[inds]) * (Sbuf[inds] - Stildebuf[inds])
        d += LA.dot(Sbuf[inds], d) / LA.dot(Sbuf[inds], Stildebuf[inds]) * (Sbuf[inds] - Stildebuf[inds])
    end

    gamma = LA.dot(stilde, Ps) / LA.dot(s, Ps)
    if abs(gamma) >= theta_bar
        theta = 1
    elseif gamma === 0 # In Julia, sign(0) = 0, whereas we define it as sign(0) = 1; handle separately
        theta = (1 - theta_bar)
    else
        theta = (1 - sign(gamma) * theta_bar) / (1 - gamma)
    end

    stilde = (1 - theta) * s + theta * stilde
    d += LA.dot(Ps, d) / LA.dot(Ps, stilde) * (s - stilde)

    if k < MAX_K
        # Update sets
        k += 1
        Sbuf[(k - 1) * n + 1 : k * n] = s
        Stildebuf[(k - 1) * n + 1 : k * n] = stilde
        PSbuf[(k - 1) * n + 1 : k * n] = Ps
    else
        k = 0
    end

    return d, k
end

function primal_dual_alg(
    x, 
    v, 
    model :: DYNAMICS_IN_L_SUPERMANN_MODEL, 
    x0 :: Vector{Float64}; 
    DEBUG :: Bool = false, 
    tol :: Float64 = 1e-12, 
    MAX_ITER_COUNT :: Int64 = 200000,
    SUPERMANN_BACKTRACKING_MAX :: Int64 = 8,
    beta :: Float64 = 0.5,
    MAX_BROYDEN_K :: Int64 = 10,
    c0 :: Float64 = 0.99,
    c1 :: Float64 = 0.99,
    q :: Float64 = 0.99,
    LOW_MEMORY :: Bool = false
)
    # Choose sigma and gamma such that sigma * gamma * model.L_norm < 1
    lambda = 0.5
    sigma = sqrt(0.99 / model.L_norm)
    gamma = sigma

    r_norm = 0
    r_safe = Inf  # Correct initial value is set during first iteration
    broyden_k = 0

    wx = zeros(length(x))
    wv = zeros(length(v))
    wxbar = zeros(length(x))
    wvbar = zeros(length(v))
    d_x = zeros(length(x))
    d_v = zeros(length(v))
    r_x = zeros(length(x))
    r_v = zeros(length(v))
    r_wx = zeros(length(x))
    r_wv = zeros(length(v))
    S = zeros((length(x) + length(v))* MAX_BROYDEN_K)
    Stilde = zeros((length(x) + length(v))* MAX_BROYDEN_K)
    PS = zeros((length(x) + length(v))* MAX_BROYDEN_K)

    d_xv = zeros(length(x) + length(v))

    xold = ones(length(x) + length(v))
    xresold = ones(length(x) + length(v))
    if !LOW_MEMORY
        H = LA.I(length(x) + length(v))
    end

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

        # Choose an update direction
        if !LOW_MEMORY
            H = broyden_sherman_morrison(H, vcat(wx, wv) - xold, vcat(wx - wxbar, wv - wvbar) - xresold, model.L, gamma, sigma)
            xold = vcat(x, v)
            xresold = vcat(x - xbar, v - vbar)
            d_x = -H[1:length(x), 1:end] * vcat(r_x, r_v)
            d_v = -H[length(x)+1 : end, 1:end] * vcat(r_x, r_v)
        else
            d_xv, broyden_k = broyden(S, Stilde, PS, d_xv, vcat(wx, wv) - xold, vcat(wx - wxbar, wv - wvbar) - xresold, vcat(x - xbar, v - vbar), broyden_k, model.L, gamma, sigma, MAX_K = MAX_BROYDEN_K)
            d_x = d_xv[1:length(x)]; d_v = d_xv[length(x) + 1 : end]
            xold = vcat(x, v)
            xresold = vcat(x - xbar, v - vbar)
        end

        # Update tau
        tau = 1

        # Educated and GKM iterations
        loop = true
        backtrack_count = 0
        while loop && backtrack_count < SUPERMANN_BACKTRACKING_MAX
            wx = x + tau * d_x
            wv = v + tau * d_v

            wxbar = wx - L_mult(model, wv * gamma, true) - Gamma_grad_mult(model, wx, gamma)
            wvbar = prox_hstar(model, x0, wv + L_mult(model, sigma * (2 * wxbar - wx)), sigma)

            r_wx = wx - wxbar
            r_wv = wv - wvbar
            rtilde_norm = sqrt(p_norm(r_wx, r_wv, r_wx, r_wv, model.L, gamma, sigma))

            # Check for educated update
            if r_norm <= r_safe && rtilde_norm <= c1 * r_norm
                copyto!(x, wx)
                copyto!(v, wv)
                r_safe = rtilde_norm + q^counter
                loop = false
                break
            end
            # Check for GKM update
            rho = p_norm(r_wx, r_wv, r_wx - tau * d_x, r_wv - tau * d_v, model.L, gamma, sigma)
            if rho >= 0.1 * r_norm * rtilde_norm
                rho = lambda * rho / rtilde_norm^2
                x += - rho * r_wx
                v += -rho * r_wv
                loop = false
                break
            end
            # Backtrack
            tau *= beta
            backtrack_count += 1
        end
        if loop === true
            # Update x by avering step
            x = lambda * xbar + (1 - lambda) * x

            # Update v by avering step
            for i = 1:length(v)
                v[i] *= (1 - lambda)
                v[i] += lambda * vbar[i]
            end
        end

        if DEBUG
            log_x[counter + 1, 1:end] = x
            log_residuals[counter + 1] = r_norm
        end

        if r_norm / sqrt(LA.norm(x)^2 + LA.norm(v)^2) < tol
            println("Breaking!", counter)
            break
        end
        counter += 1
    end

    if DEBUG
        writedlm("output/log_supermann_x.dat", log_x[1:counter, 1:end], ',')
        writedlm("output/log_supermann_residual.dat", log_residuals[1:counter], ',') 
    end

    return x
end

function solve_model(model :: DYNAMICS_IN_L_SUPERMANN_MODEL, x0 :: Vector{Float64}; tol :: Float64 = 1e-10, verbose :: Bool = false, return_all :: Bool = false, z0 :: Union{Vector{Float64}, Nothing} = nothing, v0 :: Union{Vector{Float64}, Nothing} = nothing)
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