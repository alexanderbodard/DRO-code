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

############################################################
# Solve stage
############################################################

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