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

# function broyden(Sbuf, Stildebuf, PSbuf, d, s, y, rx, k, L, alpha1, alpha2; MAX_K = 20, theta_bar = 0.5)
#     Ps = P_mult(s, L, alpha1, alpha2)
#     d = -rx
#     stilde = y
#     n = length(s)
#     for i = 1 : k
#         inds = (i - 1) * n + 1 : i * n
#         stilde += LA.dot(Sbuf[inds], stilde) / LA.dot(Sbuf[inds], Stildebuf[inds]) * (Sbuf[inds] - Stildebuf[inds])
#         d += LA.dot(Sbuf[inds], d) / LA.dot(Sbuf[inds], Stildebuf[inds]) * (Sbuf[inds] - Stildebuf[inds])
#     end

#     gamma = LA.dot(stilde, Ps) / LA.dot(s, Ps)
#     if abs(gamma) >= theta_bar
#         theta = 1
#     elseif gamma === 0 # In Julia, sign(0) = 0, whereas we define it as sign(0) = 1; handle separately
#         theta = (1 - theta_bar)
#     else
#         theta = (1 - sign(gamma) * theta_bar) / (1 - gamma)
#     end

#     stilde = (1 - theta) * s + theta * stilde
#     d += LA.dot(Ps, d) / LA.dot(Ps, stilde) * (s - stilde)

#     if k < MAX_K
#         # Update sets
#         k += 1
#         Sbuf[(k - 1) * n + 1 : k * n] = s
#         Stildebuf[(k - 1) * n + 1 : k * n] = stilde
#         PSbuf[(k - 1) * n + 1 : k * n] = Ps
#     else
#         k = 0
#     end

#     return d, k
# end

function broyden(Sbuf, Stildebuf, PSbuf, d, s, y, rx, k, L, alpha1, alpha2; MAX_K = 20, theta_bar = 0.2)
    # Ps = P_mult(s, L, alpha1, alpha2)
    d = -rx
    stilde = y
    n = length(s)
    for i = 1 : k
        inds = (i - 1) * n + 1 : i * n
        stilde += LA.dot(Sbuf[inds], stilde) * Stildebuf[inds]
        d += LA.dot(Sbuf[inds], d) * (Stildebuf[inds])
    end

    gamma = LA.dot(stilde, s) / LA.dot(s, s)
    if abs(gamma) >= theta_bar
        theta = 1
    elseif gamma === 0 # In Julia, sign(0) = 0, whereas we define it as sign(0) = 1; handle separately
        theta = (1 - theta_bar)
    else
        theta = (1 - sign(gamma) * theta_bar) / (1 - gamma)
    end

    stilde = theta / (1 - theta + theta * gamma) / LA.dot(s, s) * (s - stilde)
    d += LA.dot(s, d) * stilde

    if k < MAX_K
        # Update sets
        k += 1
        Sbuf[(k - 1) * n + 1 : k * n] = s
        Stildebuf[(k - 1) * n + 1 : k * n] = stilde
        # PSbuf[(k - 1) * n + 1 : k * n] = Ps
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
    MAX_ITER_COUNT :: Int64 = 20000,
    SUPERMANN_BACKTRACKING_MAX :: Int64 = 8,
    beta :: Float64 = 0.5,
    MAX_BROYDEN_K :: Int64 = 250,
    k2_sigma :: Float64 = 0.1,
    c0 :: Float64 = 0.99,
    c1 :: Float64 = 0.99,
    q :: Float64 = 0.99,
    LOW_MEMORY :: Bool = false
)
    # Choose sigma and gamma such that sigma * gamma * model.L_norm < 1
    lambda = 1.#0.5
    sigma = 0.99 / sqrt(model.L_norm)
    gamma = sigma

    r_norm = 0
    r_norm0 = Inf
    r_norm_old = Inf
    r_safe = Inf  # Correct initial value is set during first iteration
    eta = r_safe
    broyden_k = 0
    didk2 = false

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

    D = 1e4
    P = [[1/gamma * LA.I(length(x)) -model.L']; [-model.L 1/sigma * LA.I(length(v))]]
    PP = LA.I(length(x) + length(v))
    # PP = P
    rho = 0
    rtilde_norm = 0

    if DEBUG
        n_z = length(x)
        log_x = zeros(MAX_ITER_COUNT, n_z)
        nx = length(model.x_inds)
        nu = length(model.u_inds)
        ns = length(model.s_inds)
        ny = length(model.y_inds)
        xinit = copy(x[1:nx])
        log_residuals = zeros(MAX_ITER_COUNT)
        log_tau = zeros(MAX_ITER_COUNT)
    end

    f(z::Vector) = begin
        xxx = copy(z[1:length(wx)])
        vvv = copy(z[length(wx) + 1 : end])
        xbars = xxx - L_mult(model, vvv*gamma, true) - Gamma_grad_mult(model, xxx, gamma)
        vbars = prox_hstar(model, x0, v + L_mult(model, sigma * (2 * xbars - xxx)), sigma)
        return vcat(xxx - xbars, vvv - vbars)
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
        r_norm_old = r_norm
        r_norm = sqrt(p_norm(r_x, r_v, r_x, r_v, model.L, gamma, sigma))
        # r_norm = sqrt(LA.dot(vcat(r_x, r_v), P * vcat(r_x, r_v)))

        if counter === 0
            r_norm0 = r_norm
        end

        if DEBUG && false
            # || Tz - \frac{z_{ref} + z}{2}||
            r1 = sqrt(LA.dot(vcat(xbar - (z_ref + x) / 2, vbar - (v_ref + v) / 2), P * vcat(xbar - (z_ref + x) / 2, vbar - (v_ref + v) / 2)))
            # || z_{ref} - z ||
            r2 = sqrt(LA.dot(vcat(x, v) - vcat(z_ref, v_ref), P * (vcat(x, v) - vcat(z_ref, v_ref))))
            if r1 > 0.5 * r2
                println(r1)
                println(r2)
                println(gamma * sigma * model.L_norm < 1)
                println(x)
                println(v)
                error("Small circle condition violated in iteration $(counter)")
            end
        end

        if didk2 && r_norm > r_norm_old && false
            println("K2 update increased our norm!: Iteration $(counter), difference is $(r_norm - r_norm_old)")
        end
        # if didk2 && p_norm(x - z_ref, v - v_ref, x - z_ref, v - v_ref, model.L, gamma, sigma) <= p_norm(xold[1:length(x)] - z_ref, xold[length(x)+1:end] - v_ref, xold[1:length(x)] - z_ref, xold[length(x)+1:end] - v_ref, model.L, gamma, sigma) - k2_sigma * r_norm_old^2
        #     println("Should happen: Iteration $(counter), difference is $(p_norm(xold[1:length(x)] - z_ref, xold[length(x)+1:end] - v_ref, xold[1:length(x)] - z_ref, xold[length(x)+1:end] - v_ref, model.L, gamma, sigma) - k2_sigma * r_norm_old^2 - p_norm(x - z_ref, v - v_ref, x - z_ref, v - v_ref, model.L, gamma, sigma))")
        # elseif didk2
        #     println("NOT GOOD! Counter: $(counter), $(p_norm(x - z_ref, v - v_ref, x - z_ref, v - v_ref, model.L, gamma, sigma)) and $(p_norm(xold[1:length(x)] - z_ref, xold[length(x)+1:end] - v_ref, xold[1:length(x)] - z_ref, xold[length(x)+1:end] - v_ref, model.L, gamma, sigma) ) and $(k2_sigma * r_norm_old^2)")
        #     println("--> difference is $(p_norm(xold[1:length(x)] - z_ref, xold[length(x)+1:end] - v_ref, xold[1:length(x)] - z_ref, xold[length(x)+1:end] - v_ref, model.L, gamma, sigma) - k2_sigma * r_norm_old^2 - p_norm(x - z_ref, v - v_ref, x - z_ref, v - v_ref, model.L, gamma, sigma))")
        # end
        didk2 = false

        # Choose an update direction
        if !LOW_MEMORY
            H = broyden_sherman_morrison(H, vcat(wx, wv) - xold, vcat(wx - wxbar, wv - wvbar) - xresold, model.L, gamma, sigma)
            xold = vcat(x, v)
            xresold = vcat(x - xbar, v - vbar)
            d_xv = -H * vcat(r_x, r_v)
            d_x = d_xv[1:length(x)]; d_v = d_xv[length(x)+1 : end]
            d_norm = sqrt(p_norm(d_x, d_v, d_x, d_v, model.L, gamma, sigma))
            if d_norm > D * r_norm
                d_x = D * (r_norm / d_norm) * d_x
                d_v = D * (r_norm / d_norm) * d_v
            end
            # d_x = -H[1:length(x), 1:end] * vcat(r_x, r_v)
            # d_v = -H[length(x)+1 : end, 1:end] * vcat(r_x, r_v)
        else
            # println(x, v)
            # jacob = ForwardDiff.jacobian(f, vcat(copy(x), copy(v)))
            # jacob[LA.diagind(jacob)] .+= 0.1
            # try
            #     d_xv = jacob \ vcat(-r_x, -r_v)
            #     # println(d_xv)
            # catch e
            #     println(e)
            #     d_xv, broyden_k = broyden(S, Stilde, PS, d_xv, vcat(wx, wv) - xold, vcat(wx - wxbar, wv - wvbar) - xresold, vcat(x - xbar, v - vbar), broyden_k, model.L, gamma, sigma, MAX_K = MAX_BROYDEN_K)
            # end
            
            d_xv, broyden_k = broyden(S, Stilde, PS, d_xv, vcat(wx, wv) - xold, vcat(wx - wxbar, wv - wvbar) - xresold, vcat(x - xbar, v - vbar), broyden_k, model.L, gamma, sigma, MAX_K = MAX_BROYDEN_K)
            
            d_x = d_xv[1:length(x)]; d_v = d_xv[length(x) + 1 : end]
            xold = vcat(x, v)
            xresold = vcat(x - xbar, v - vbar)

            # println(x)
        end

        # d_x = zeros(length(d_x)); d_v = zeros(length(d_v))

        loop = true
        backtrack_count = 0

        # if r_norm <= c0 * eta
        #     eta = r_norm
        #     x += d_x
        #     v += d_v


        #     wx = copy(x)
        #     wv = copy(v)
        #     wxbar = wx - L_mult(model, wv * gamma, true) - Gamma_grad_mult(model, wx, gamma)
        #     wvbar = prox_hstar(model, x0, wv + L_mult(model, sigma * (2 * wxbar - wx)), sigma)

        #     loop = false
        # end

        # Update tau
        tau = 1

        # Educated and GKM iterations
        while loop && backtrack_count < SUPERMANN_BACKTRACKING_MAX
            wx = x + tau * d_x
            wv = v + tau * d_v

            wxbar = wx - L_mult(model, wv * gamma, true) - Gamma_grad_mult(model, wx, gamma)
            wvbar = prox_hstar(model, x0, wv + L_mult(model, sigma * (2 * wxbar - wx)), sigma)
            # println("Iteration $(counter)")

            r_wx = wx - wxbar
            r_wv = wv - wvbar
            rtilde_norm = sqrt(p_norm(r_wx, r_wv, r_wx, r_wv, model.L, gamma, sigma))

            if DEBUG && false
                # || Tz - \frac{z_{ref} + z}{2}||
                r1 = sqrt(LA.dot(vcat(wxbar - (z_ref + wx) / 2, wvbar - (v_ref + wv) / 2), P * vcat(wxbar - (z_ref + wx) / 2, wvbar - (v_ref + wv) / 2)))
                # || z_{ref} - z ||
                r2 = sqrt(LA.dot(vcat(wx, wv) - vcat(z_ref, v_ref), P * (vcat(wx, wv) - vcat(z_ref, v_ref))))
                if r1 > 0.5 * r2
                    println(r1)
                    println(r2)
                    println(gamma * sigma * model.L_norm < 1)
                    println(r_wx)
                    println(r_wv)
                    error("Small circle condition violated by W in iteration $(counter)")
                end
            end

            # Check for educated update
            # println("$(r_safe - r_norm) and $(c1 * r_norm - rtilde_norm)")
            if r_norm <= r_safe && rtilde_norm <= c1 * r_norm
                println("Iteration $(counter) has tau=$(tau)")
                copyto!(x, wx)
                copyto!(v, wv)
                r_safe = rtilde_norm + q^counter# * r_norm0
                loop = false
                break
            end
            # Check for GKM update
            rho = p_norm(r_wx, r_wv, r_wx - tau * d_x, r_wv - tau * d_v, model.L, gamma, sigma)
            # rho = LA.dot(vcat(r_wx, r_wv), vcat(r_wx, r_wv)) - LA.dot(vcat(r_wx, r_wv), vcat(wx - x, wv - v))
            # rho = LA.dot(vcat(r_wx, r_wv), P * vcat(r_wx, r_wv)) - LA.dot(vcat(r_wx, r_wv), P * vcat(tau * d_x, tau * d_v))
            # rho = p_norm(r_wx, r_wv, x - wxbar, v - wvbar, model.L, gamma, sigma)
            # println("--------------")
            # println("$(rho) and $(k2_sigma * r_norm * rtilde_norm)")
            # println("$(LA.dot(vcat(r_wx, r_wv), vcat(r_wx - tau * d_x, r_wv - tau * d_v))) and $(k2_sigma * LA.norm(vcat(r_wx, r_wv)) * LA.norm(vcat(r_x, r_v)))")

            if rho >= k2_sigma * r_norm * rtilde_norm
                if DEBUG
                    z1 = vcat(x, v)
                    z2 = rand(length(x) + length(v))
                    znorm = sqrt(p_dot(z1 - z2, z1 - z2, PP))

                    w = vcat(wx, wv)

                    cw = vcat(r_wx, r_wv)
                    cw_norm_squared = p_dot(cw, cw, PP)
                    bw = p_dot(cw, w, PP) - cw_norm_squared
                    
                    proj1 = z1 - (p_dot(z1, cw, PP) - bw) / cw_norm_squared * cw
                    proj2 = z2 - (p_dot(z2, cw, PP) - bw) / cw_norm_squared * cw
                    proj_norm = sqrt(p_dot(proj1 - proj2, proj1 - proj2, PP))
                    if proj_norm > znorm
                        println("---\nProjection is expansive in iteration $(counter).\nProjection norm: $(proj_norm), z_norm: $(znorm)\nShould project? $(p_dot(z, cw, PP) > bw), <c, z> = $(p_dot(z, cw, PP)), bw = $(bw)")
                        println("Should be equal if projection worked: $(p_dot(cw, proj, PP)) versus $(bw)")
                    else
                        temp = lambda / 2 * z1 + (1 - lambda / 2) * proj1
                        # println(temp[1:5])
                    end
                end

                didk2 = true
                # println("--------------")
                # println("$(rho) and $(k2_sigma * r_norm * rtilde_norm)")
                # println("$(LA.dot(vcat(r_wx, r_wv), P * vcat(r_wx, r_wv)) - LA.dot(vcat(r_wx, r_wv), P * vcat(wx - x, wv - v)))")
                # println("Iteration $(counter) has GKM update with tau = $(tau)")
                # rho = lambda * rho / rtilde_norm^2

                rho /= rtilde_norm^2
                x = x - lambda * rho * r_wx
                v = v - lambda * rho * r_wv
                # println(x[1:5])
                # error()
                loop = false
                break
            end
            # Backtrack
            tau *= beta
            backtrack_count += 1
        end
        if loop === true
            println("Iteration $(counter) has reached max")
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
            log_tau[counter+1] = tau

            # println("R norm: ", r_norm)

            if counter > 0 && didk2 && p_dot(vcat(x, v), vcat(x, v), P) > p_dot(xold, xold, P) && false
                println("This really should not happen")
            end

            if counter > 0 && r_norm > log_residuals[counter] && false
                println(rho)
                println(rtilde_norm)
                # println(LA.norm(vcat(r_wx, r_wv)))
                println(p_norm(r_wx, r_wv, tau * d_x, tau * d_v, model.L, gamma, sigma))
                error("Should not happen: $(counter), $(r_norm) and $(log_residuals[counter])")
            end
        end

        if r_norm < tol * sqrt(LA.norm(x)^2 + LA.norm(v)^2)
            println("Breaking!", counter)
            break
        end
        counter += 1
    end

    if DEBUG
        println("Writing outputs")
        writedlm("output/log_supermann_x.dat", log_x[1:counter, 1:end], ',')
        writedlm("output/log_supermann_residual.dat", log_residuals[1:counter], ',') 
        writedlm("output/log_supermann_tau.dat", log_tau[1:counter], ',') 
    end

    return x
end

function solve_model(model :: DYNAMICS_IN_L_SUPERMANN_MODEL, x0 :: Vector{Float64}; tol :: Float64 = 1e-8, verbose :: Bool = false, return_all :: Bool = false, z0 :: Union{Vector{Float64}, Nothing} = nothing, v0 :: Union{Vector{Float64}, Nothing} = nothing)
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