include("custom_model.jl")

"""
Performs a bisection method.

Func must be callable.
g_lb and g_ub will be altered by calling this function.
"""
function bisection_method!(g_lb, g_ub, tol, psi)
    if psi(g_lb)*psi(g_ub) > 0
        error("Incorrect initial interval. Found $(psi(g_lb)) and $(psi(g_ub))")
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

function build_h_x_model(scen_tree :: ScenarioTree, rms :: Vector{Riskmeasure})
    eliminate_states = false
    n_z = get_n_z(scen_tree, rms, eliminate_states)
    z = zeros(n_z)

    n_L = get_n_L(scen_tree, rms, eliminate_states)
    L = construct_L(scen_tree, rms, n_L, n_z)
    L_trans = L'

    proj = z -> begin
        # 4a
        offset = 0
        for k = 1:scen_tree.n_non_leaf_nodes
            n_z_part = size(rms[k].A)[2]
            ind = collect(offset + 1 : offset + n_z_part)
            z[ind] = MOD.projection_on_set(MOD.DefaultDistance(), z[ind], rms[k].C.subcones[1]) # TODO: Fix polar cone

            offset += n_z_part
        end

        # 4b
        for k = 1:scen_tree.n_non_leaf_nodes
            n_z_part = length(rms[k].b)
            ind = collect(offset + 1 : offset + n_z_part)
            z[ind] = MOD.projection_on_set(MOD.DefaultDistance(), z[ind], rms[k].D.subcones[1]) # TODO: Fix polar cone

            offset += n_z_part
        end

        # 4c
        for k = 1:scen_tree.n_non_leaf_nodes
            n_z_part = length(rms[k].b) + 1
            ind = collect(offset + 1 : offset + n_z_part)
            
            b_bar = [1; rms[k].b]
            dot_p = LA.dot(z[ind], b_bar)
            if dot_p > 0
                z[ind] = dot_p / LA.dot(b_bar, b_bar) .* b_bar
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
        R_offset = scen_tree.n * scen_tree.n_x
        T = length(scen_tree.min_index_per_timestep)
        Q_bars = []
        # Q_bar is a block diagonal matrix with the corresponding Q's and R's for that scenario
        for scen_ind = 1:length(scenarios)
            scenario = scenarios[scen_ind]

            L_I = Float64[]
            L_J = Float64[]
            L_V = Float64[]

            for t = 1:T
                nn = scenario[t]
                # Add Q to Q_bar
                Q_I, Q_J, Q_V = findnz(sparse(cost.Q[t]))

                append!(L_I, Q_I .+ (nn - 1) * scen_tree.n_x)
                append!(L_J, Q_J .+ (nn - 1) * scen_tree.n_x)
                append!(L_V, 2 .* Q_V)

                if t < T
                    # Add R to Q_bar
                    R_I, R_J, R_V = findnz(sparse(cost.R[t]))

                    append!(L_I, R_I .+ R_offset .+ (nn - 1) * scen_tree.n_u)
                    append!(L_J, R_J .+ R_offset .+ (nn - 1) * scen_tree.n_u)
                    append!(L_V, 2 .* R_V)
                end
            end

            append!(Q_bars, [sparse(L_I, L_J, L_V, 
                scen_tree.n * scen_tree.n_x + scen_tree.n_non_leaf_nodes * n_u, 
                scen_tree.n * scen_tree.n_x + scen_tree.n_non_leaf_nodes * n_u
            )])
        end
        # Compute projection
        for scen_ind  = 1:length(scenarios)
            n_z_part = scen_tree.n * scen_tree.n_x + scen_tree.n_non_leaf_nodes * n_u
            ind = collect(offset + 1 : offset + n_z_part)
            z_temp = z[ind]
            s = z[n_z_part + offset + 1]

            f = z_temp' * Q_bars[scen_ind] * z_temp
            if f > s
                prox_f = gamma -> begin
                    I, J, V = findnz(Q_bars[scen_ind])
                    n_Q_x, n_Q_y = size(Q_bars[scen_ind])
                    M_temp = sparse(I, J, 1 ./ (V .+ gamma), n_Q_x, n_Q_y)
                    M_temp * (z_temp ./ gamma)
                end

                psi = gamma -> begin
                    temp = prox_f(gamma)

                    temp' * Q_bars[scen_ind] * temp - gamma - s
                end
                local g_lb = 1e-12 # TODO: How close to zero?
                local g_ub = 1.
                gamma_star = bisection_method!(g_lb, g_ub, 1e-4, psi)
                println(gamma_star)
                z[ind], z[n_z_part + offset + 1] = prox_f(gamma_star), s + gamma_star
            end

            offset += n_z_part + 1
        end

        # 4e: Dynamics
        n_z_part = scen_tree.n_x * (scen_tree.n - 1)
        ind = collect(offset + 1 : offset + n_z_part)
        z[ind] = zeros(n_z_part)

        return z
    end

    return CustomModel(
        z -> L * z,
        z -> L_trans * z,
        x -> (
            temp = zeros(length(x));
            temp[1] = 1;
            temp
        ),
        (z, gamma) -> begin
            z - gamma * proj(z / gamma)
        end
    )
end