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

    n_y_start_index = (scen_tree.n_non_leaf_nodes * scen_tree.n_u  # Inputs
                        + n_x * scen_tree.n                        # State at t=0
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
        ind = collect(
        (k - 1) * n_y + 1 : k * n_y
        )
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

        for _ = length(scen_tree.min_index_per_timestep)-1:-1:1
            n = scen_tree.anc_mapping[k]
            pushfirst!(xxs, xx[node_to_x(scen_tree, n)]...)
            pushfirst!(uus, uu[node_to_u(scen_tree, n)]...)
        end
        println(typeof(xxs))
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
        append!(L_J, AJ .+ (scen_tree.n_u * (anc_node - 1)) .+ u_offset)
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
function construct_L(scen_tree :: ScenarioTree, rms :: Vector{Riskmeasure}, n_L :: Int64, n_z :: Int64)
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
    append!(L_V, ones(length(scen_tree.n_x)))

    println(n_L, ", ", maximum(L_I))

    return sparse(L_I, L_J, L_V, n_L, n_z)
end

"""
Performs a bisection method.

Func must be callable.
g_lb and g_ub will be altered by calling this function.
"""
function bisection_method!(g_lb, g_ub, tol, psi)
    # while psi(g_lb)*psi(g_ub) > 0
    #     g_ub *= 2
    # end

    # println(g_ub)

    if ( psi(g_lb) + tol ) * ( psi(g_ub) - tol ) > 0 # only work up to a precision of the tolerance
        error("Incorrect initial interval. Found $(psi(g_lb)) and $(psi(g_ub)) which results in $(( psi(g_lb) + tol ) * ( psi(g_ub) - tol ))")
    end

    while abs(g_ub-g_lb) > tol
        g_new = (g_lb + g_ub) / 2.
        if psi(g_lb) * psi(g_new) < 0
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

function primal_dual_alg(x, v, model :: DYNAMICS_IN_L_MODEL)
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
    sigma = 1e-2
    gamma = 1e-2
    sigma = 0.4
    gamma = 0.4
    Sigma = sigma * sparse(LA.I(n_L))
    Gamma = gamma * sparse(LA.I(n_z))
    lambda = 0.5

    if (sigma * gamma * model.L_norm > 1)
        error("sigma and gamma are not chosen correctly")
    end

    plot_vector = zeros(20000, 11)
    x0 = copy(x[1:3])

    # TODO: Work with some tolerance
    counter = 0
    while counter < 20000
        x_old = copy(x) # TODO: Check Julia behaviour

        xbar = x - Gamma * model.Ltrans(v) - Gamma * model.grad_f(x)
        vbar = model.prox_hstar_Sigmainv(v + Sigma * model.L(2 * xbar - x), 1 ./ sigma, counter % 100 == 0)

        # if counter == 1
        #     println(vbar)
        # end

        x = lambda * xbar + (1 - lambda) * x
        v = lambda * vbar + (1 - lambda) * v

        plot_vector[counter + 1, 1:end] = x

        # if counter < 3
        #     println("-----------")
        #     println("x: ", x[z_to_x(scen_tree)])
        #     println("u: ", x[z_to_u(scen_tree)])
        #     println("s: ", x[z_to_s(scen_tree)])
        #     println("y: ", x[z_to_y(scen_tree, 4)])
        #     println("v: ", v)
        # end

        # println("Solution norm: ", LA.norm((x - x_old) / x) / length(x))
        if LA.norm((x - x_old)) / LA.norm(x) < 1e-12
            println("Breaking!", counter)
            break
        end
        counter += 1
    end

    residues = Float64[]
    for i = 1:counter
        append!(residues, LA.norm(plot_vector[i, 1:3] .- x_ref) / LA.norm(x0 .- x_ref))
    end
    fixed_point_residues = Float64[]
    for i = 2:counter
        append!(fixed_point_residues, LA.norm(plot_vector[i, 1:3] .- plot_vector[i-1, 1:3]) / LA.norm(plot_vector[i, 1:3]))
    end

    pgfplotsx()
    plot(log10.(collect(1:length(fixed_point_residues))), log10.(fixed_point_residues), fmt = :png, xlims = (0, 1 * log10(length(fixed_point_residues))), labels=["fixed_point_residue_x"])
    # plot!(plot_vector[1:counter, 1:3], fmt = :png, labels=["x"])
    filename = "fixed_point_residue_x.png"
    savefig(filename)

    plot(log10.(collect(1:length(residues))), log10.(residues), fmt = :png, xlims = (0, 1 * log10(length(residues))), labels=["residue_x"])
    # plot!(plot_vector[1:counter, 1:3], fmt = :png, labels=["x"])
    filename = "debug_x_res.png"
    savefig(filename)

    plot(plot_vector[1:counter, 1:3], fmt = :png, labels=["x"])
    # plot!(plot_vector[1:counter, 1:3], fmt = :png, labels=["x"])
    filename = "debug_x.png"
    savefig(filename)

    plot(plot_vector[1:counter, 4], fmt = :png, labels=["u"])
    filename = "debug_u.png"
    savefig(filename)

    plot(plot_vector[1:counter, 5:7], fmt = :png, labels=["s"])
    filename = "debug_s.png"
    savefig(filename)

    plot(plot_vector[1:counter, 8:11], fmt = :png, labels=["y"])
    filename = "debug_y.png"
    savefig(filename)

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
    L = construct_L(scen_tree, rms, n_L, n_z)
    L_trans = L'

    proj = (z, log) -> begin
        # 4a
        offset = 0
        for k = 1:scen_tree.n_non_leaf_nodes
            n_z_part = size(rms[k].A)[2]
            ind = collect(offset + 1 : offset + n_z_part)
            z[ind] = MOD.projection_on_set(MOD.DefaultDistance(), z[ind], MOI.Nonpositives(2)) # TODO: Fix polar cone
            # println("4a: ", z[ind])

            offset += n_z_part
        end

        # 4b
        for k = 1:scen_tree.n_non_leaf_nodes
            n_z_part = length(rms[k].b)
            ind = collect(offset + 1 : offset + n_z_part)
            # println("Before: ", z[ind])
            z[ind] = MOD.projection_on_set(MOD.DefaultDistance(), z[ind], MOI.Nonpositives(4)) # TODO: Fix polar cone
            # println("Should be negative: ", z[ind])
            offset += n_z_part
        end

        # 4c
        for k = 1:scen_tree.n_non_leaf_nodes
            n_z_part = length(rms[k].b) + 1
            ind = collect(offset + 1 : offset + n_z_part)
            
            b_bar = [1; rms[k].b]
            dot_p = LA.dot(z[ind], b_bar)
            if dot_p > 0
                z[ind] = z[ind] - dot_p / LA.dot(b_bar, b_bar) .* b_bar
            end

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

            append!(Q_bars, [sparse(L_I, L_J, L_V, 
                length(scen_tree.min_index_per_timestep) * scen_tree.n_x + (length(scen_tree.min_index_per_timestep) - 1) * n_u, 
                length(scen_tree.min_index_per_timestep) * scen_tree.n_x + (length(scen_tree.min_index_per_timestep) - 1) * n_u
            )])
        end
        # display(Q_bars)

        # Compute projection
        for scen_ind  = 1:length(scenarios)
            n_z_part = length(scen_tree.min_index_per_timestep) * scen_tree.n_x + (length(scen_tree.min_index_per_timestep) - 1) * n_u
            ind = collect(offset + 1 : offset + n_z_part)
            z_temp = z[ind]
            s = z[n_z_part + offset + 1]

            f = 0.5 * z_temp' * Q_bars[scen_ind] * z_temp
            # if (log)
            #     println("z_temp: ", z_temp, ", f: ", f, ", s: ", s)
            # end
            if f > s
                prox_f = gamma -> begin
                    I, J, V = findnz(Q_bars[scen_ind])
                    n_Q_x, n_Q_y = size(Q_bars[scen_ind])
                    M_temp = sparse(I, J, 1 ./ (V .+ (1 ./ gamma)), n_Q_x, n_Q_y)
                    M_temp * (z_temp ./ gamma)
                end

                psi = gamma -> begin
                    temp = prox_f(gamma)

                    0.5 * temp' * Q_bars[scen_ind] * temp - gamma - s
                end

                # println("f: ", f)
                # println("prox: ", prox_f(f)' * Q_bars[scen_ind] * prox_f(f))
                # println("psi: ", psi(f))
                # println("s: ", s)

                local g_lb = 1e-12 # TODO: How close to zero?
                local g_ub = f - s #1. TODO: Can be tighter with gamma
                gamma_star = bisection_method!(g_lb, g_ub, 1e-8, psi)
                # println("s + gamma_star: ", s + gamma_star)
                # println(gamma_star)
                z[ind], z[n_z_part + offset + 1] = prox_f(gamma_star), s + gamma_star
            end

            # ppp, sss = epigraph_qcqp(Q_bars[scen_ind], z_temp, s)
            # if log
            #     println("ppp, sss:", ppp, ", ", sss)
            #     println("custom: ", z[ind], ", ", z[n_z_part + offset + 1])
            # end
            # z[ind], z[n_z_part + offset + 1] = ppp, sss

            offset += (n_z_part + 1)
        end

        # 4e: Dynamics
        n_z_part = scen_tree.n_x * (scen_tree.n - 1)
        ind = collect(offset + 1 : offset + n_z_part)
        z[ind] = zeros(n_z_part)

        # Initial condition
        z[end] = 2.

        return z
    end

    L_norm = maximum(LA.svdvals(collect(L)))^2
    println(L_norm)

    return DYNAMICS_IN_L_MODEL(
        z -> L * z,
        z -> L_trans * z,
        x -> begin
            temp = zeros(length(x));
            temp[z_to_s(scen_tree)[1]] = 1;
            temp
        end,
        (z, gamma, log) -> begin
            # if log
            #     println("-----------------")
            #     println("Projection: ", gamma * proj(z / gamma, false)[12:21])
            #     println("z: ", z[12:21] / gamma)
            #     println("full z", z / gamma)
            #     # println("At index 12: ", z[12])
            #     # println((gamma * proj(z / gamma))[12])
            # end
            res = z - proj(z * gamma, log) / gamma
            # res[end] = z[end] - 2.
            res
        end,
        L_norm

    )
end

function solve_model(model :: DYNAMICS_IN_L_MODEL, x0 :: Vector{Float64}, tol :: Float64 = 1e-8)
    n_z = get_n_z(scen_tree, rms, false)
    n_L = get_n_L(scen_tree, rms, false)

    z = zeros(n_z)
    # z[1:3] = x_ref
    # z[4] = u_ref[1]
    # z[5:7] = s_ref
    # z[8:11] = y_ref
    v = zeros(n_L)

    z = primal_dual_alg(z, v, model)

    println("x: ", z[
        1:scen_tree.n * scen_tree.n_x
    ])

    println("u: ", z[z_to_u(scen_tree)])

    println("s: ", z[z_to_s(scen_tree)])

    println("y: ", z[z_to_y(scen_tree, 4)])

    return z[
        1:scen_tree.n * scen_tree.n_x
    ], z[z_to_u(scen_tree)] 
end