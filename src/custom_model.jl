############################################################
# Build stage
############################################################

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

function get_n_L(scen_tree :: ScenarioTree, rms :: Vector{Riskmeasure}, eliminate_states :: Bool; RICATTI :: Bool = false)
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
    if !RICATTI
        n_dynamics = (scen_tree.n - 1) * scen_tree.n_x
    else
        n_dynamics = scen_tree.n * scen_tree.n_x + (n_timesteps - 1)
    end
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
function construct_L_with_dynamics(scen_tree :: ScenarioTree, rms :: Vector{Riskmeasure}, dynamics :: Dynamics, n_L :: Int64, n_z :: Int64)
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
Currently doesn't support elimination of states
"""
function construct_L_ricatti(scen_tree :: ScenarioTree, rms :: Vector{Riskmeasure}, dynamics :: Dynamics, n_L :: Int64, n_z :: Int64)
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

    L_II, L_JJ, L_VV = findnz(sparse(LA.I(scen_tree.n * scen_tree.n_x + (length(scen_tree.min_index_per_timestep) - 1))))
    append!(L_I, L_II .+ maximum(L_I))
    append!(L_J, L_JJ)
    append!(L_V, L_VV)

    # Initial condition
    append!(L_I, maximum(L_I) .+ collect(1:scen_tree.n_x))
    append!(L_J, collect(1:scen_tree.n_x))
    append!(L_V, ones(scen_tree.n_x))
    
    return sparse(L_I, L_J, L_V, n_L, n_z)
end

############################################################
# Solve stage
############################################################

function prox_f_copy(Q, gamma, x)
    return x ./ (Q .+ 1. / gamma) / gamma
end

function prox_f!(Q, gamma, x, output)
    @simd for i = 1:length(x)
        @inbounds @fastmath output[i] = x[i] / (Q[i] + 1. / gamma) / gamma
    end
end

function prox_f!(Q, gamma, x)
    @simd for i = 1:length(x)
        @inbounds @fastmath x[i] = x[i] / (Q[i] + 1. / gamma) / gamma
    end
end

function psi_copy(Q, gamma, x, s)
    temp = prox_f_copy(Q, gamma, x)
    return 0.5 * sum(Q .* temp.^2) - gamma - s
end

function psi!(Q, gamma, x, s)
    prox_f!(Q, gamma, x)
    res = 0
    @simd for i = 1:length(x)
        @inbounds @fastmath res += Q[i] * x[i]^2
    end
    return 0.5 * res - gamma - s
end

function psi!(Q, gamma, x, s, workspace)
    prox_f!(Q, gamma, x, workspace)
    res = 0
    @simd for i = 1:length(x)
        @inbounds @fastmath res += Q[i] * workspace[i]^2
    end
    return 0.5 * res - gamma - s
end

function bisection_method_copy!(g_lb, g_ub, tol, Q, x, s)
    g_new = (g_lb + g_ub) / 2
    ps = psi_copy(Q, g_new, x, s)
    while abs(g_ub - g_lb) > tol
        if sign(ps) > 0
            g_lb = g_new
        elseif sign(ps) < 0
            g_ub = g_new
        else
            return g_new
            error("Should not happen")
        end
        g_new = (g_lb + g_ub) / 2
        ps = psi_copy(Q, g_new, x, s)
    end
    return g_new
end

"""
x must not be changed when returning!
TODO: Does this function actually change some arguments?
"""
function bisection_method!(g_lb, g_ub, tol, Q, x, s)
    g_new = (g_lb + g_ub) / 2
    xcopy = copy(x)
    ps = psi!(Q, g_new, x, s, xcopy)
    while abs(g_ub - g_lb) > tol
        if sign(ps) > 0
            g_lb = g_new
        elseif sign(ps) < 0
            g_ub = g_new
        else
            # copyto!(x, xcopy)
            return g_new
            error("Should not happen")
        end
        g_new = (g_lb + g_ub) / 2
        # copyto!(x, xcopy)
        ps = psi!(Q, g_new, x, s, xcopy)
    end
    # copyto!(x, xcopy)
    return g_new
end

function epigraph_bisection(Q, x, t)
    f = 0.5 * sum(Q .* (x.^2))
    if f > t
        local g_lb = 0 # TODO: How close to zero?
        local g_ub = f - t #1. TODO: Can be tighter with gamma
        tol = 1e-12
        gamma_star = bisection_method_copy!(g_lb, g_ub, tol, Q, x, t)
        res = prox_f_copy(Q, gamma_star, x)
        # return res, 0.5 * sum(res .* Q .* res)
        return prox_f_copy(Q, gamma_star, x), t + gamma_star #* ( 1 + tol )
    end
    return x, t
end

function epigraph_bisection!(Q, x, t)
    f = 0.5 * sum(Q .* (x.^2))
    if f > t
        local g_lb = 0 # TODO: How close to zero?
        local g_ub = f - t #1. TODO: Can be tighter with gamma
        tol = 1e-12
        gamma_star = bisection_method!(g_lb, g_ub, tol, Q, x, t)
        prox_f!(Q, gamma_star, x)
        return t + gamma_star
    end
    return t
end

function epigraph_qcqp(Q, x, t)
    if 0.5 * x' * Q * x <= t
        return x, t
    end

    model = Model(Mosek.Optimizer)
    set_silent(model)

    @variable(model, p[i=1:length(x)])
    @variable(model, s)

    @objective(model, Min, (p[1] - x[1])^2 + (p[2] - x[2])^2 + (s - t)^2)

    @constraint(model, 0.5 * p' * Q * p - s <= 0)

    optimize!(model)
    return value.(model[:p]), value.(model[:s])
end

function L_mult(model :: DYNAMICS_IN_L_MODEL, z :: Vector, transp :: Bool = false)
    return transp ? model.L' * z : model.L * z
end

function Gamma_grad_mult(model :: DYNAMICS_IN_L_MODEL, z :: Vector, gamma :: Float64)
    temp = zeros(length(z));
    temp[model.s_inds[1]] = gamma;
    return temp
end

function projection(model :: DYNAMICS_IN_L_MODEL, x0 :: Vector{Float64}, z :: Vector)
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

        z[ind], z[ind[end] + 1] = epigraph_bisection(model.Q_bars[scen_ind], z_temp, s)

        # ppp, sss = epigraph_qcqp(LA.diagm(model.Q_bars[scen_ind]), z_temp, s)
        # z[ind], z[ind[end] + 1] = ppp, sss
    end

    # 4e: Dynamics
    z[model.inds_4e] = zeros(length(model.inds_4e))

    # Initial condition
    z[end - length(x0) + 1 : end] = x0

    return z
end

function prox_hstar(model :: DYNAMICS_IN_L_MODEL, x0 :: Vector{Float64}, z :: Vector, gamma :: Float64)
    return z - projection(model, x0, z / gamma) * gamma
end

function p_norm(ax, av, bx, bv, L, gamma, sigma)
    return 1 / gamma * LA.dot(ax, bx) - ax' * L' * bv - av' * L * bx + 1 / sigma * LA.dot(av, bv)
end

function p_dot(a, b, P)
    return LA.dot(a, P*b)
end

function dot_p(a, b, L, alpha1, alpha2)
    n2, n1 = size(L)
    ax = a[1:n1]; av = a[n1+1:end]
    bx = b[1:n1]; bv = b[n1+1:end]
    return p_norm(ax, av, bx, bv, L, alpha1, alpha2)
end