include("custom_model.jl")

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

function build_h_x_model(scen_tree :: ScenarioTree, rms :: Vector{Riskmeasure})
    eliminate_states = false
    n_z = get_n_z(scen_tree, rms, eliminate_states)
    z = zeros(n_z)

    n_L = get_n_L(scen_tree, rms, eliminate_states)
    L = construct_L(scen_tree, rms, n_L, n_z)
    # display(collect(L)[1:2, 1:end])
    # display(collect(L)[3:6, 1:end])
    # display(collect(L)[7:11, 1:end])
    # display(collect(L)[12:19, 1:end])
    # display(collect(L)[20:22, 1:end])
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

    return CustomModel(
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