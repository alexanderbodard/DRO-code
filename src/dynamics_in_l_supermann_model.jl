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
    Q_bars = Vector{Float64}[]
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
        zeros(n_L),
        zeros(n_L),
        zeros(n_L),
        zeros(n_z),
        zeros(n_L),
        zeros(n_z),
        zeros(n_L),
        zeros(n_z),
        zeros(n_L),
        zeros(scen_tree.n_x)
    )
end

############################################################
# Solve stage
############################################################

function broyden_sherman_morrison(H, delta_z, delta_R, L, alpha1, alpha2; theta_bar = 0.2)
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

# function broyden_sherman_morrison(H, delta_z, delta_R, L, alpha1, alpha2; theta_bar = 0.5)
#     gamma = dot_p(H * delta_R, delta_z, L, alpha1, alpha2) / dot_p(delta_z, delta_z, L, alpha1, alpha2)
#     if abs(gamma) >= theta_bar
#         theta = 1
#     elseif gamma === 0 # In Julia, sign(0) = 0, whereas we define it as sign(0) = 1; handle separately
#         theta = (1 - theta_bar)
#     else
#         theta = (1 - sign(gamma) * theta_bar) / (1 - gamma)
#     end

#     s_tilde = (1 - theta) * delta_z + theta * H * delta_R # With powell

#     return H + 1 / dot_p(delta_z, s_tilde, L, alpha1, alpha2) * (delta_z - (s_tilde)) * (delta_z' * H)
# end

function P_mult(x, L, alpha1, alpha2)
    x1 = x[1:size(L)[2]]; x2 = x[size(L)[2] + 1 : end]

    return vcat(x1 / alpha1 - L' * x2, -L * x1 + x2 / alpha2)
end

# function broyden(Sbuf, Stildebuf, PSbuf, d, s, y, rx, k, L, alpha1, alpha2; MAX_K = 20, theta_bar = 0.2)
#   # Ps = P_mult(s, L, alpha1, alpha2)
#   d = -rx
#   stilde = y
#   n = length(s)
#   for i = 1 : k
#       inds = (i - 1) * n + 1 : i * n
#       stilde += LA.dot(Sbuf[inds], stilde) * Stildebuf[inds]
#       d += LA.dot(Sbuf[inds], d) * (Stildebuf[inds])
#   end

#   gamma = LA.dot(stilde, s) / LA.dot(s, s)
#   if abs(gamma) >= theta_bar
#       theta = 1
#   elseif gamma === 0 # In Julia, sign(0) = 0, whereas we define it as sign(0) = 1; handle separately
#       theta = (1 - theta_bar)
#   else
#       theta = (1 - sign(gamma) * theta_bar) / (1 - gamma)
#   end

#   stilde = theta / (1 - theta + theta * gamma) / LA.dot(s, s) * (s - stilde)
#   d += LA.dot(s, d) * stilde

#   if k < MAX_K
#       # Update sets
#       k += 1
#       Sbuf[(k - 1) * n + 1 : k * n] = s
#       Stildebuf[(k - 1) * n + 1 : k * n] = stilde
#       # PSbuf[(k - 1) * n + 1 : k * n] = Ps
#   else
#       k = 0
#   end

#   return d, k
# end

function broyden(Sbuf, Stildebuf, PSbuf, d, s, y, rx, k, L, alpha1, alpha2; MAX_K = 20, theta_bar = 0.5)
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

function primal_dual_alg!(
    model :: DYNAMICS_IN_L_SUPERMANN_MODEL;
    tol :: Float64 = 1e-8, 
    MAX_ITER_COUNT :: Int64 = 100000,
    SUPERMANN_BACKTRACKING_MAX :: Int64 = 8,
    beta :: Float64 = 0.5,
    MAX_BROYDEN_K :: Int64 = 20,
    k2_sigma :: Float64 = 0.1,
    c0 :: Float64 = 0.99,
    c1 :: Float64 = 0.99,
    q :: Float64 = 0.99,
    LOW_MEMORY :: Bool = true,
    verbose :: VERBOSE_LEVEL = SILENT,
    path = "logs/",
    filename  = "logs",
    log_stride :: Int64 = 1
)
    # Choose sigma and gamma such that sigma * gamma * model.L_norm < 1
    lambda = 1.#0.5
    sigma = 0.99 / sqrt(model.L_norm)
    gamma = sigma

    r_norm = 0
    r_norm0 = 0
    r_safe = Inf  # Correct initial value is set during first iteration
    eta = r_safe
    broyden_k = 0

    # x = model.z
    v = model.v

    wx = zeros(model.nz)
    wv = zeros(model.nv)
    wxbar = zeros(model.nz)
    wvbar = zeros(model.nv)
    d_x = zeros(model.nz)
    d_v = zeros(model.nv)
    r_x = zeros(model.nz)
    r_v = zeros(model.nv)
    r_wx = zeros(model.nz)
    r_wv = zeros(model.nv)
    S = zeros((model.nz + model.nv)* MAX_BROYDEN_K)
    Stilde = zeros((model.nz + model.nv)* MAX_BROYDEN_K)
    PS = zeros((model.nz + model.nv)* MAX_BROYDEN_K)

    d_xv = zeros(model.nz + model.nv)

    xold = ones(model.nz + model.nv)
    xresold = ones(model.nz + model.nv)
    if !LOW_MEMORY
        H = LA.I(model.nz + model.nv)
    end

    D = 1e4
    P = [[1/gamma * LA.I(model.nz) -model.L']; [-model.L 1/sigma * LA.I(model.nv)]]
    rho = 0
    rtilde_norm = 0

    # Preallocate extra memory for logging 
    if verbose == PRINT_AND_WRITE
      println("Starting solve step...")

      nx = length(model.x_inds)
      n_iter_log = Int(floor(MAX_ITER_COUNT * SUPERMANN_BACKTRACKING_MAX / log_stride))
      rnorms = zeros(n_iter_log)
      xs = zeros(n_iter_log, nx)
      ks = zeros(n_iter_log)
    end

    # TODO: Work with some tolerance
    counter = 0
    log_counter = 0
    while counter < MAX_ITER_COUNT
        if counter === 1
          r_norm0 = r_norm
        end

        # Compute xbar
        copyto!(model.v_workspace, v)
        for i = 1:model.nv
            model.v_workspace[i] *= gamma
        end
        xbar = model.z - L_mult(model, model.v_workspace, true) - Gamma_grad_mult(model, model.z, gamma)

        # Compute vbar
        for i = 1:model.nz
            model.z_workspace[i] = sigma * (2 * xbar[i] - model.z[i])
        end

        vbar = prox_hstar(model, model.x0, v + L_mult(model, model.z_workspace), sigma)

        # Compute the residual
        r_x = model.z - xbar
        r_v = v - vbar
        r_norm = sqrt(p_norm(r_x, r_v, r_x, r_v, model.L, gamma, sigma))

        # if verbose == PRINT_AND_WRITE && counter % log_stride == 0
        #   rnorms[counter ÷ log_stride + 1] = r_norm
        #   xs[(counter ÷ log_stride +1), :] = model.z[model.x_inds]
        # end

        # Choose an update direction
        if !LOW_MEMORY
            H = broyden_sherman_morrison(H, vcat(wx, wv) - xold, vcat(wx - wxbar, wv - wvbar) - xresold, model.L, gamma, sigma)
            xold = vcat(model.z, v)
            xresold = vcat(model.z - xbar, v - vbar)
            d_xv = -H * vcat(r_x, r_v)
            d_x = d_xv[1:model.nz]; d_v = d_xv[model.nz+1 : end]
        else            
            d_xv, broyden_k = broyden(S, Stilde, PS, d_xv, vcat(wx, wv) - xold, vcat(wx - wxbar, wv - wvbar) - xresold, vcat(model.z - xbar, v - vbar), broyden_k, model.L, gamma, sigma, MAX_K = MAX_BROYDEN_K)
            d_x = d_xv[1:model.nz]; d_v = d_xv[model.nz + 1 : end]
            xold = vcat(model.z, v)
            xresold = vcat(model.z - xbar, v - vbar)
        end

        d_norm = sqrt(p_norm(d_x, d_v, d_x, d_v, model.L, gamma, sigma))
        if d_norm > D * r_norm
            d_x = D * (r_norm / d_norm) * d_x
            d_v = D * (r_norm / d_norm) * d_v
        end

        loop = true
        backtrack_count = 0

        # Blind update
        if r_norm <= c0 * eta && false
            eta = r_norm
            x += d_x
            v += d_v

            wx = copy(x)
            wv = copy(v)
            wxbar = wx - L_mult(model, wv * gamma, true) - Gamma_grad_mult(model, wx, gamma)
            wvbar = prox_hstar(model, model.x0, wv + L_mult(model, sigma * (2 * wxbar - wx)), sigma)

            loop = false
        end

        # Update tau
        tau = 1

        # Educated and GKM iterations
        while loop && backtrack_count < SUPERMANN_BACKTRACKING_MAX
            wx = model.z + tau * d_x
            wv = v + tau * d_v

            wxbar = wx - L_mult(model, wv * gamma, true) - Gamma_grad_mult(model, wx, gamma)
            wvbar = prox_hstar(model, model.x0, wv + L_mult(model, sigma * (2 * wxbar - wx)), sigma)

            r_wx = wx - wxbar
            r_wv = wv - wvbar
            rtilde_norm = sqrt(p_norm(r_wx, r_wv, r_wx, r_wv, model.L, gamma, sigma))

            log_counter += 1

            if verbose == PRINT_AND_WRITE && log_counter % log_stride == 0
              rnorms[log_counter ÷ log_stride + 1] = r_norm
              xs[(log_counter ÷ log_stride +1), :] = model.z[model.x_inds]
            end

            # Check for educated update
            if r_norm <= r_safe && rtilde_norm <= c1 * r_norm
                copyto!(model.z, wx)
                copyto!(v, wv)
                r_safe = rtilde_norm + q^counter
                loop = false

                if verbose == PRINT_AND_WRITE && counter % log_stride == 0
                  ks[counter ÷ log_stride + 1] = tau
                end

                break
            end
            # Check for GKM update
            rho = p_norm(r_wx, r_wv, r_wx - tau * d_x, r_wv - tau * d_v, model.L, gamma, sigma)
            if rho >= k2_sigma * r_norm * rtilde_norm
                rho /= rtilde_norm^2
                # model.z = model.z - lambda * rho * r_wx
                for i = 1:model.nz
                  model.z[i] -= lambda * rho * r_wx[i]
                end
                v = v - lambda * rho * r_wv
                loop = false

                if verbose == PRINT_AND_WRITE && counter % log_stride == 0
                  ks[counter ÷ log_stride + 1] = - tau
                end

                break
            end
            # Backtrack
            tau *= beta
            backtrack_count += 1
        end
        if loop === true
            # Update x by averaging step
            # model.z = lambda * xbar + (1 - lambda) * model.z
            for i = 1:model.nz
              model.z[i] *= (1 - lambda)
              model.z[i] += lambda * xbar[i]
            end

            # Update v by averaging step
            for i = 1:model.nv
                v[i] *= (1 - lambda)
                v[i] += lambda * vbar[i]
            end
        end

        if r_norm < tol * r_norm0
            println("Breaking!", counter)
            break
        end
        counter += 1
    end

    # Write away logs
    if verbose == PRINT_AND_WRITE
      println("Writing logs to output file...")

      writedlm(path * filename * "_residual.dat", rnorms[1:log_counter ÷ log_stride], ',')
      writedlm(path * filename * "_x.dat", xs[1:log_counter ÷ log_stride, :], ',')
      writedlm(path * filename * "_ks.dat", ks[1:counter ÷ log_stride, :], ',')
      println("Finished logging.")
    end
end

function solve_model(
  model :: DYNAMICS_IN_L_SUPERMANN_MODEL, 
  x0 :: Vector{Float64}; 
  tol :: Float64 = 1e-8, 
  verbose :: VERBOSE_LEVEL = SILENT,
  path = "logs/",
  filename  = "logs",
  log_stride :: Int64 = 1,
  return_all :: Bool = false, 
  z0 :: Union{Vector{Float64}, Nothing} = nothing, 
  v0 :: Union{Vector{Float64}, Nothing} = nothing, 
  MAX_ITER_COUNT :: Int64 = 100000,
  LOW_MEMORY :: Bool = true,
)
    if z0 !== nothing && v0 !== nothing
        copyto!(model.z, z0)
        copyto!(model.v, v0)
    end

    copyto!(model.x0, x0)

    primal_dual_alg!(model, tol=tol, MAX_ITER_COUNT=MAX_ITER_COUNT, verbose=verbose, filename = filename, path=path, log_stride=log_stride, LOW_MEMORY = LOW_MEMORY)
end