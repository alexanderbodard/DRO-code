function get_n_z(scen_tree :: ScenarioTree, rms :: Vector{Riskmeasure}, eliminate_states :: Bool)
    if eliminate_states
        n_x = 0                             # Eliminate state variables
    else
        n_x = scen_tree.n * scen_tree.n_x   # Every node has a state
    end

    return ((length(scen_tree.min_index_per_timestep) - 1) * scen_tree.n_u # One input per time step
                + n_x                                               # n_x
                + scen_tree.n                                       # s variable: 1 component per node
                + scen_tree.n_non_leaf_nodes * length(rms[1].b))    # One y variable for each non leaf node
end

function get_n_L(scen_tree :: ScenarioTree, rms :: Vector{Riskmeasure}, eliminate_states :: Bool)
    n_y = length(rms[1].b)
    if eliminate_states
        n_x = 0                             # Eliminate state variables
    else
        n_x = scen_tree.n * scen_tree.n_x   # Every node has a state
    end

    n_timesteps = length(scen_tree.min_index_per_timestep)

    n_L_rows = (size(rms[1].A)[2]    # 4a
                + n_y                # 4b
                + 1 + n_y)           # 4c
    n_cost_i = (scen_tree.n_x * n_timesteps    # x
        + (n_timesteps - 1) * scen_tree.n_u    # u
        + 1)                                   # x_{T-1}[i]
    n_cost = n_cost_i * (scen_tree.n - scen_tree.n_non_leaf_nodes)
    n_dynamics = (scen_tree.n - 1) * scen_tree.n_x
    return scen_tree.n_non_leaf_nodes * n_L_rows + n_cost + n_dynamics + scen_tree.n_x # Initial condition!
end

"""
Currently doesn't support elimination of states
"""
function construct_L_4a(scen_tree :: ScenarioTree, rms :: Vector{Riskmeasure}, n_z :: Int64, n_y :: Int64)
    L_I = Float64[]
    L_J = Float64[]
    L_V = Float64[]

    n_y_start_index = ((length(scen_tree.min_index_per_timestep) - 1) * scen_tree.n_u  # Inputs
                        + scen_tree.n_x * scen_tree.n              # State at t=0
                        + scen_tree.n                              # S variables
                        + 1)

    ss = z_to_s(scen_tree)
    yy = z_to_y(scen_tree, n_y)
    for k = 1:scen_tree.n_non_leaf_nodes
        Is = Float64[]
        Js = Float64[]
        Vs = Float64[]

        Js = ss[scen_tree.child_mapping[k]]
        append!(Is, [i for i in collect(1:length(Js))])
        append!(Vs, [1 for _ in 1:length(Js)])
        S_s = sparse(Is, Js, Vs, length(Is), n_y_start_index - 1)

        Is = Float64[]
        Js = Float64[]
        Vs = Float64[]

        Js = yy[(k - 1) * n_y + 1 : k * n_y] .- (n_y_start_index - 1)
        append!(Is, [i for i in collect(1:length(Js))])
        append!(Vs, [1 for _ in 1:length(Js)])
        S_y = sparse(Is, Js, Vs, length(Is), n_z - n_y_start_index + 1)

        r = rms[k]
        S = hcat(sparse(r.A') * S_s, sparse(r.B') * S_y)
        SI, SJ, SV = findnz(S)
        if k > 1
            append!(L_I, SI .+ maximum(L_I))
        else
            append!(L_I, SI)
        end
        append!(L_J, SJ)
        append!(L_V, SV)
    end

    return L_I, L_J, L_V
end

"""
Currently doesn't support elimination of states
"""
function construct_L_4b(scen_tree :: ScenarioTree, n_y :: Int64)
    L_II = Float64[]
    L_JJ = Float64[]
    L_VV = Float64[]

    yy = z_to_y(scen_tree, n_y)
    for k = 1:scen_tree.n_non_leaf_nodes
        ind = (k - 1) * n_y + 1 : k * n_y
        append!(L_JJ, yy[ind])
    end
    append!(L_II, [i for i in collect(1 : scen_tree.n_non_leaf_nodes * n_y)])
    append!(L_VV, [1 for _ in 1:scen_tree.n_non_leaf_nodes * n_y])
    
    return L_II, L_JJ, L_VV
end

"""
Currently doesn't support elimination of states
"""
function construct_L_4c(scen_tree :: ScenarioTree, n_y :: Int64)
    L_III = Float64[]
    L_JJJ = Float64[]
    L_VVV = Float64[]

    yy = z_to_y(scen_tree, n_y)
    ss = z_to_s(scen_tree)
    for k = 1:scen_tree.n_non_leaf_nodes
        append!(L_JJJ, ss[k])
        ind = collect(
        (k - 1) * n_y + 1 : k * n_y
        )
        append!(L_JJJ, yy[ind])
    end
    append!(L_III, [i for i in collect(1 : scen_tree.n_non_leaf_nodes * (n_y + 1))])
    append!(L_VVV, [-1 for _ in 1:scen_tree.n_non_leaf_nodes * (n_y + 1)])

    return L_III, L_JJJ, L_VVV
end

"""
Currently doesn't support elimination of states
"""
function construct_L_4d(scen_tree :: ScenarioTree)
    L_I = Float64[]
    L_J = Float64[]
    L_V = Float64[]

    xx = z_to_x(scen_tree)
    uu = z_to_u(scen_tree)
    ss = z_to_s(scen_tree)
    for k = scen_tree.leaf_node_min_index : scen_tree.leaf_node_max_index
        xxs = xx[node_to_x(scen_tree, k)]
        uus = []
        sss = [ss[k]]

        n = k
        for _ = length(scen_tree.min_index_per_timestep)-1:-1:1
            n = scen_tree.anc_mapping[n]
            pushfirst!(xxs, xx[node_to_x(scen_tree, n)]...)
            pushfirst!(uus, uu[node_to_timestep(scen_tree, n)]...)
        end
        append!(L_J, xxs)
        append!(L_J, uus)
        append!(L_J, sss)
    end
    append!(L_I, [i for i in 1 : length(L_J)])
    append!(L_V, [1 for _ in 1 : length(L_J)])

    return L_I, L_J, L_V
end

"""
Impose dynamics H * z = 0 directly. H is included in the L matrix
"""
function construct_L_4e(scen_tree :: ScenarioTree, dynamics :: Dynamics, n_z :: Int64)
    L_I = Float64[]
    L_J = Float64[]
    L_V = Float64[]

    u_offset = scen_tree.n * scen_tree.n_x
    I_offset = 0
    # J_offset = 0
    B_J_offset = 0
    for k = 2 : scen_tree.n
        # For every non-root state, impose dynamics
        w = scen_tree.node_info[k].w
        A = dynamics.A[w]; B = dynamics.B[w]
        I = LA.I(size(A)[1])
        anc_node = scen_tree.anc_mapping[k]

        # x+ = A x + B u

        # A
        AI, AJ, AV = findnz(A)
        append!(L_I, AI .+ I_offset)
        append!(L_J, AJ .+ (scen_tree.n_x * (anc_node - 1)))
        append!(L_V, AV)

        # -I
        AI, AJ, AV = findnz(-I)
        append!(L_I, AI .+ I_offset)
        append!(L_J, AJ .+ (scen_tree.n_x) * (k - 1))
        append!(L_V, AV)

        # B
        AI, AJ, AV = findnz(B)
        append!(L_I, AI .+ I_offset)
        append!(L_J, AJ .+ (scen_tree.n_u * (node_to_timestep(scen_tree, anc_node) - 1)) .+ u_offset)
        append!(L_V, AV)

        I_offset += size(A)[1]
        # J_offset += size(A)[2] # TODO: A is always square, so can be done with a single offset?
        # B_J_offset += size(B)[2]
    end

    return L_I, L_J, L_V
end

"""
Currently doesn't support elimination of states
"""
function construct_L(scen_tree :: ScenarioTree, rms :: Vector{Riskmeasure}, dynamics :: Dynamics, n_L :: Int64, n_z :: Int64)
    n_y = length(rms[1].b)

    L_I, L_J, L_V = construct_L_4a(scen_tree, rms, n_z, n_y)
    L_II, L_JJ, L_VV = construct_L_4b(scen_tree, n_y)
    L_III, L_JJJ, L_VVV = construct_L_4c(scen_tree, n_y)
    L_IIII, L_JJJJ, L_VVVV = construct_L_4d(scen_tree)

    append!(L_I, L_II .+ maximum(L_I))
    append!(L_I, L_III .+ maximum(L_I))
    append!(L_I, L_IIII .+ maximum(L_I))
    append!(L_J, L_JJ, L_JJJ, L_JJJJ)
    append!(L_V, L_VV, L_VVV, L_VVVV)

    if 1 == 1 # TODO, add option to choose whether we include this
        L_II, L_JJ, L_VV = construct_L_4e(scen_tree, dynamics, n_z)

        append!(L_I, L_II .+ maximum(L_I))
        append!(L_J, L_JJ)
        append!(L_V, L_VV)
    end

    # Initial condition
    append!(L_I, maximum(L_I) .+ collect(1:scen_tree.n_x))
    append!(L_J, collect(1:scen_tree.n_x))
    append!(L_V, ones(scen_tree.n_x))

    return sparse(L_I, L_J, L_V, n_L, n_z)
end

"""
Performs a bisection method.

Func must be callable.
g_lb and g_ub will be altered by calling this function.
"""
function bisection_method!(g_lb, g_ub, tol, Q, z_temp, s, workspace_vec)
    # while psi(g_lb)*psi(g_ub) > 0
    #     g_ub *= 2
    # end

    if ( psi(Q, g_lb, z_temp, s, workspace_vec) + tol ) * ( psi(Q, g_ub, z_temp, s, workspace_vec) - tol ) > 0 # only work up to a precision of the tolerance
        error("Incorrect initial interval. Found $(psi(Q, g_lb, z_temp, s, workspace_vec)) and $(psi(Q, g_ub, z_temp, s, workspace_vec)) which results in $(( psi(Q, g_lb, z_temp, s, workspace_vec) + tol ) * ( psi(Q, g_ub, z_temp, s, workspace_vec) - tol ))")
    end

    while abs(g_ub-g_lb) > tol
        g_new = (g_lb + g_ub) / 2.
        if psi(Q, g_lb, z_temp, s, workspace_vec) * psi(Q, g_new, z_temp, s, workspace_vec) < 0
            g_ub = g_new
        else
            g_lb = g_new
        end
    end
    return (g_lb + g_ub) / 2.
end

function epigraph_qcqp(Q, x, t)
    model = Model(Mosek.Optimizer)
    set_silent(model)

    @variable(model, p[i=1:length(x)])
    @variable(model, s)
    # @variable(model, tt >= 0)

    # @objective(model, Min, sum((p[i] - x[i])^2 for i = 1:length(x)) + sum((s - t)^2))
    # @objective(model, Min, [tt; vcat(p, s) - vcat(x, t)] in SecondOrderCone())
    @objective(model, Min, (p[1] - x[1])^2 + (p[2] - x[2])^2 + (s - t)^2)

    @constraint(model, 0.5 * p' * Q * p - s <= 0)

    optimize!(model)
    return value.(model[:p]), value.(model[:s])
end

function L_mult(model :: DYNAMICS_IN_L_MODEL, z :: Vector{Float64}, transp :: Bool = false)
    transp ? model.L' * z : model.L * z
end

function Gamma_grad_mult(model :: DYNAMICS_IN_L_MODEL, z :: Vector{Float64}, gamma :: Float64)
    temp = zeros(length(z));
    temp[model.s_inds[1]] = 1;
    return gamma .* temp
end

function prox_f(Q, gamma, z_temp, workspace_vec)
    copyto!(workspace_vec, z_temp)
    for i = 1:length(z_temp)
        workspace_vec[i] /= gamma * (Q[i] + 1. / gamma)
    end
    return workspace_vec
end

function psi(Q, gamma, z_temp, s, workspace_vec)
    workspace_vec = prox_f(Q, gamma, z_temp, workspace_vec)

    # return 0.5 * temp' * Q * temp - gamma - s
    # return 0.5 * sum(Q .* temp.^2) - gamma - s
    res = - gamma - s
    for i = 1:length(Q)
        res += 0.5 * Q[i] * workspace_vec[i]^2
    end
    return res
end

function projection(model :: DYNAMICS_IN_L_MODEL, x0 :: Vector{Float64}, z :: Vector{Float64})
    # 4a
    for ind in model.inds_4a
        z[ind] = MOD.projection_on_set(MOD.DefaultDistance(), z[ind], MOI.Nonpositives(2)) # TODO: Fix polar cone
    end

    # 4b
    z[model.inds_4b] = MOD.projection_on_set(MOD.DefaultDistance(), z[model.inds_4b], MOI.Nonpositives(4)) # TODO: Fix polar cone
    

    # 4c
    for (i, ind) in enumerate(model.inds_4c)
        b_bar = model.b_bars[i]
        dot_p = LA.dot(z[ind], b_bar)
        if dot_p > 0
            z[ind] = z[ind] - dot_p / LA.dot(b_bar, b_bar) .* b_bar
        end
    end

    # 4d: Compute projection
    for (scen_ind, ind) in enumerate(model.inds_4d)
        z_temp = z[ind]
        s = z[ind[end] + 1]

        # f = 0.5 * sum(model.Q_bars[scen_ind] .* (z_temp.^2))
        f = 0
        for i = 1:length(z_temp)
            model.workspace_vec[i] = z_temp[i]
            model.workspace_vec[i] *= z_temp[i]
            model.workspace_vec[i] *= model.Q_bars[scen_ind][i]
            f += model.workspace_vec[i]
        end
        f *= 0.5
        if f > s
            local g_lb = 1e-8 # TODO: How close to zero?
            local g_ub = f - s #1. TODO: Can be tighter with gamma
            gamma_star = bisection_method!(g_lb, g_ub, 1e-8, model.Q_bars[scen_ind], z_temp, s, model.workspace_vec)
            z[ind], z[ind[end] + 1] = prox_f(model.Q_bars[scen_ind], gamma_star, z_temp, model.workspace_vec), s + gamma_star
        end

        # ppp, sss = epigraph_qcqp(Q_bars[scen_ind], z_temp, s)
        # z[ind], z[ind[end] + 1] = ppp, sss
    end

    # 4e: Dynamics
    z[model.inds_4e] = zeros(length(model.inds_4e))

    # Initial condition
    z[end - length(x0) + 1 : end] = x0

    return z
end

function prox_hstar(model :: DYNAMICS_IN_L_MODEL, x0 :: Vector{Float64}, z :: Vector{Float64}, gamma :: Float64)
    return z - projection(model, x0, z / gamma) * gamma
end

# """
# Closely follows SuperMann paper Algorithm 3

# - k is the number of elements in S and Stilde when calling this function

# """
# function restarted_broyden!(
#     S :: Vector{Float64}, 
#     Stilde :: Vector{Float64}, 
#     s :: Vector{Float64},
#     s_tilde :: Vector{Float64},
#     y :: Vector{Float64}, 
#     rx :: Vector{Float64},
#     d :: Vector{Float64},
#     k :: Int64; 
#     MAX_K :: Int64 = 20,
#     theta_bar :: Float64 = 0.5
# )
#     nrx = length(rx)

#     # Initialize d and s_tilde
#     copyto!(d, -rx)
#     copyto!(s_tilde, y)

#     # Loop over the given sets S and Stilde
#     if k >= 1
#         for i = 1:k
#             inds = (i - 1) * nrx + 1 : i * nrx
#             s_tilde += LA.dot(S[inds], s_tilde) * Stilde[inds]
#             d += LA.dot(S[inds], d) * Stilde[inds]
#         end
#     end

#     # Compute theta
#     gamma = LA.dot(s_tilde, s) / LA.norm(s)^2
#     if abs(gamma) >= theta_bar
#         theta = 1
#     elseif gamma === 0 # In Julia, sign(0) = 0, whereas we define it as sign(0) = 1; handle separately
#         theta = (1 - theta_bar)
#     else
#         theta = (1 - sign(gamma) * theta_bar) / (1 - gamma)
#     end

#     # Compute final s_tilde and d
#     s_tilde = theta / (1 - theta + theta * gamma) / LA.norm(s)^2 * (s - s_tilde)
#     d += LA.dot(s, d) * s_tilde

#     # Update sets S and S_tilde and counter k
#     if k === MAX_K
#         k = 0
#         # Clear sets
#         # Well, let's not actually do this but just reset the index
#     else
#         k += 1
#         # Update sets
#         S[(k - 1) * nrx + 1 : k * nrx] = s
#         Stilde[(k - 1) * nrx + 1 : k * nrx] = s_tilde
#     end

#     return d
# end

function p_norm(ax, av, bx, bv, L, alpha1, alpha2)
    return 1 / alpha1 * LA.dot(ax, bx) - ax' * L' * bv - av' * L * bx + 1 / alpha2 * LA.dot(av, bv)
end

function dot_p(a, b, L, alpha1, alpha2)
    n2, n1 = size(L)
    ax = a[1:n1]; av = a[n1+1:end]
    bx = b[1:n1]; bv = b[n1+1:end]
    return p_norm(ax, av, bx, bv, L, alpha1, alpha2)
end

function sherman_morrison(H, delta_z, delta_R, L, alpha1, alpha2; theta_bar = 0.5)
    gamma = LA.dot(H * delta_R, delta_z) / LA.norm(delta_z)^2
    if abs(gamma) >= theta_bar
        theta = 1
    elseif gamma === 0 # In Julia, sign(0) = 0, whereas we define it as sign(0) = 1; handle separately
        theta = (1 - theta_bar)
    else
        theta = (1 - sign(gamma) * theta_bar) / (1 - gamma)
    end

    # y_tilde = (1 - theta) * (H \ delta_z) + theta * delta_R
    # s_tilde = H * y_tilde
    s_tilde = (1 - theta) * delta_z + theta * H * delta_R # With powell
    # s_tilde = H * delta_R # without Powell

    # return H + 1 / dot_p(delta_z, s_tilde, L, alpha1, alpha2) * (delta_z - (s_tilde)) * (delta_z' * H)
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
    end

    return d
end

function primal_dual_alg(
    x, 
    v, 
    model :: DYNAMICS_IN_L_MODEL, 
    x0 :: Vector{Float64}; 
    DEBUG :: Bool = false, 
    tol :: Float64 = 1e-12, 
    MAX_ITER_COUNT :: Int64 = 200000,
    SUPERMANN :: Bool = true,
    SUPERMANN_BACKTRACKING_MAX :: Int64 = 8,
    beta :: Float64 = 0.5,
    MAX_BROYDEN_K :: Int64 = 10,
    c0 :: Float64 = 0.99,
    c1 :: Float64 = 0.99,
    q :: Float64 = 0.99
)
    n_z = length(x)
    n_L = length(v)

    # TODO: Initialize Sigma and Gamma in some way
    # Currently they are defined as c * LA.I, so that the bisection
    # method can be used to retrieve c
    """
    - Sigma in S_{++}^{n_L}
    - Gamma in S_{++}^{n_z}

    Choose sigma and gamma such that
    sigma * gamma * model.L_norm < 1
    """
    lambda = 0.9
    sigma = sqrt(0.9 / model.L_norm)
    sigma_inv = 1 / sigma
    gamma = sigma

    if (sigma * gamma * model.L_norm > 1)
        error("sigma and gamma are not chosen correctly")
    end

    if DEBUG
        plot_vector = zeros(MAX_ITER_COUNT, n_z)
        nx = length(model.x_inds)
        nu = length(model.u_inds)
        ns = length(model.s_inds)
        ny = length(model.y_inds)
        xinit = copy(x[1:nx])
        residuals = zeros(MAX_ITER_COUNT)
    end

    x_workspace = copy(x)
    v_workspace = copy(v)

    if SUPERMANN

        r_norm = 0
        # eta = 0     # Correct initial value is set during first iteration
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
        # stilde_workspace = zeros(length(x) + length(v))

        xold = ones(length(x) + length(v))
        xresold = ones(length(x) + length(v))
        # H = LA.I(length(x) + length(v))
    end

    # TODO: Work with some tolerance
    counter = 0
    while counter < MAX_ITER_COUNT

        # Compute xbar
        copyto!(v_workspace, v)
        for i = 1:length(v)
            v_workspace[i] *= gamma
        end
        xbar = x - L_mult(model, v_workspace, true) - Gamma_grad_mult(model, x, gamma)

        # Compute vbar
        for i = 1:length(x)
            x_workspace[i] = sigma * (2 * xbar[i] - x[i])
        end

        vbar = prox_hstar(model, x0, v + L_mult(model, x_workspace), sigma)

        # Compute the residual
        r_x = x - xbar
        r_v = v - vbar
        r_norm = sqrt(p_norm(r_x, r_v, r_x, r_v, model.L, gamma, sigma))

        # Update
        if SUPERMANN
            """
            This implementation closely follows Algorithm 2 of the SuperMann paper.
            """
            # Choose an update direction

            # Restarted Broyden but in one call (should be like this I think?)
            # d_xv = restarted_broyden!(
            #     S,
            #     Stilde,
            #     vcat(wx, wv) - xold,
            #     stilde_workspace,
            #     vcat(wx - wxbar, wv - wvbar) - xresold,
            #     vcat(r_x, r_v),
            #     d_xv,
            #     broyden_k,
            #     MAX_K = MAX_BROYDEN_K,
            #     theta_bar = 0.5
            # )
            # d_x = d_xv[1:length(x)]
            # d_v = d_xv[length(x) + 1 : end]
            # xold = vcat(x, v)
            # xresold = vcat(x - xbar, v - vbar)

            # H = sherman_morrison(H, vcat(wx, wv) - xold, vcat(wx - wxbar, wv - wvbar) - xresold, model.L, gamma, sigma)
            # xold = vcat(x, v)
            # xresold = vcat(x - xbar, v - vbar)
            # d_x = -H[1:length(x), 1:end] * vcat(r_x, r_v)
            # d_v = -H[length(x)+1 : end, 1:end] * vcat(r_x, r_v)

            # println(d_x)

            d_xv = broyden(S, Stilde, PS, d_xv, vcat(wx, wv) - xold, vcat(wx - wxbar, wv - wvbar) - xresold, vcat(x - xbar, v - vbar), broyden_k, model.L, gamma, sigma, MAX_K = MAX_BROYDEN_K)
            d_x = d_xv[1:length(x)]; d_v = d_xv[length(x) + 1 : end]
            xold = vcat(x, v)
            xresold = vcat(x - xbar, v - vbar)

            if broyden_k === MAX_BROYDEN_K
                broyden_k = 0
            else
                broyden_k += 1
            end

            # Update tau (eta remains unchanged)
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
                # rtilde_norm = sqrt(LA.norm(r_wx)^2 + LA.norm(r_wv)^2)
                rtilde_norm = sqrt(p_norm(r_wx, r_wv, r_wx, r_wv, model.L, gamma, sigma))

                # println("rtilde_norm: $(rtilde_norm), r_norm: $(r_norm), r_safe = $(r_safe)")

                # Check for educated update
                if r_norm <= r_safe && rtilde_norm <= c1 * r_norm
                    # println("Educated update with tau = $(tau) in iteration $(counter)!")
                    copyto!(x, wx)
                    copyto!(v, wv)
                    r_safe = rtilde_norm + q^counter
                    # println("Educated update")
                    loop = false
                    break
                end
                # Check for GKM update
                # rho = LA.dot(vcat(r_wx, r_wv), vcat(r_wx, r_wv) - tau * vcat(d_x, d_v))
                rho = p_norm(r_wx, r_wv, r_wx - tau * d_x, r_wv - tau * d_v, model.L, gamma, sigma)
                if rho >= 0.1 * r_norm * rtilde_norm
                    # println("GKM update in iteration $(counter)!")
                    # println("Rho = $(rho), must be larger than $(0.1 * r_norm * rtilde_norm)")
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
                # println("Maximum number of backtracking attained. Nominal update in iteration $(counter)")
                # Update x by avering step
                x = lambda * xbar + (1 - lambda) * x

                # Update v by avering step
                for i = 1:length(v)
                    v[i] *= (1 - lambda)
                    v[i] += lambda * vbar[i]
                end
            end
            if loop === false
                # println("Supermann in action in iteration $(counter)!")
            end
        else
            """
            Vanilla CP
            """
            # Update x by avering step
            x = lambda * xbar + (1 - lambda) * x

            # Update v by avering step
            for i = 1:length(v)
                v[i] *= (1 - lambda)
                v[i] += lambda * vbar[i]
            end
        end

        if DEBUG
            plot_vector[counter + 1, 1:end] = x
            residuals[counter + 1] = r_norm
        end

        if r_norm / LA.norm(x) < tol
            println("Breaking!", counter)
            break
        end
        counter += 1
    end

    if DEBUG        
        residues = Float64[]
        for i = 1:counter
            append!(residues, LA.norm(plot_vector[i, 1:length(model.x_inds)] .- x_ref) / LA.norm(xinit .- x_ref))
        end
        plot(collect(1:length(residues)), log10.(residues), fmt = :png, labels=["Vanilla"])
    end

    if DEBUG
        residues = Float64[]
        for i = 1:counter
            append!(residues, LA.norm(plot_vector[i, 1:length(model.x_inds)] .- x_ref) / LA.norm(xinit .- x_ref))
        end
        fixed_point_residues = Float64[]
        for i = 2:counter
            append!(fixed_point_residues, LA.norm(plot_vector[i, 1:length(model.x_inds)] .- plot_vector[i-1, 1:length(model.x_inds)]) / LA.norm(plot_vector[i, 1:length(model.x_inds)]))
        end

        pgfplotsx()
        # plot(collect(1:length(fixed_point_residues)), log10.(fixed_point_residues), fmt = :png, xlims = (0, 1 * length(fixed_point_residues)), labels=["fixed_point_residue_x"])
        # filename = "fixed_point_residue_x.png"
        # savefig(filename)

        plot!(collect(1:length(residues)), log10.(residues), fmt = :png, labels=["SuperMann"])
        filename = "debug_x_res.png"
        savefig(filename)

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
    
        # # Check whether the constraints are correctly imposed
        # println("--------")
        # println("Verifying 4a - 4c: Risk epigraph constraints...")
        # H = model.L[
        #     1 : (size(rms[1].A)[2] + length(rms[1].b)) * scen_tree.n_non_leaf_nodes, 1:end
        # ]
        # println("Should be smaller than zero: ", maximum(H * x))

        # for i = 1:scen_tree.n_non_leaf_nodes
        #     z_temp = vcat(-x[nx + nu + 1 : nx + nu + ns][i], -x[nx + nu + ns + 1 : nx + nu + ns + ny][(i - 1) * length(rms[1].b) + 1 : i * length(rms[1].b)])            
        #     b_bar = model.b_bars[i]
        #     dot_p = LA.dot(z_temp, b_bar)
        #     if dot_p > 0
        #         println("4c has been violated, dot product equals $(dot_p) at iteration $(i)!!!")
        #         println("B_bar equals $(b_bar), x_bar equals $(z_temp)")
        #     end
        # end

        # println("--------")
        # println("Verifying 4d: cost")


        # println("--------")
        # println("Verifying 4e: dynamics...")
        # H = model.L[
        #     end - nx + 1: end - length(x0), 1:nx + nu
        # ]

        # println("Dynamics matrix structure: ", H)
        # println(maximum(H * x[1:nx+nu]))
        # println(maximum(H * vcat(x_ref, u_ref)))

        # println("-------")
    end

    return x
end

##########
# Refactor
##########

function build_dynamics_in_l_model(scen_tree :: ScenarioTree, cost :: Cost, dynamics :: Dynamics, rms :: Vector{Riskmeasure})
    eliminate_states = false
    n_z = get_n_z(scen_tree, rms, eliminate_states)
    z = zeros(n_z)

    n_L = get_n_L(scen_tree, rms, eliminate_states)
    L = construct_L(scen_tree, rms, dynamics, n_L, n_z)
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

    return DYNAMICS_IN_L_MODEL(
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
        )
    )
end

function solve_model(model :: DYNAMICS_IN_L_MODEL, x0 :: Vector{Float64}; tol :: Float64 = 1e-10, verbose :: Bool = false, return_all :: Bool = false, z0 :: Union{Vector{Float64}, Nothing} = nothing, v0 :: Union{Vector{Float64}, Nothing} = nothing, SUPERMANN :: Bool = true)
    z = zeros(model.nz)
    v = zeros(model.nv)

    if z0 !== nothing && v0 !== nothing
        z = z0
        v = v0
    end

    z = primal_dual_alg(z, v, model, x0, tol=tol, DEBUG=verbose, SUPERMANN=SUPERMANN)

    println(model.L_norm)

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