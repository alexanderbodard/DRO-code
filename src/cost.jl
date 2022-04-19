###
# Cost
###

"""
Struct defining the stage cost function at each timestep:
    The cost is assumed quadratic at each time step
    l_i (x, u) = x_i' * Q_i  * x_i + u_i' * R_i * u_i
    - Q: Vector of Q_i matrices
    - R: Vector of R_i matrices
"""
struct Cost
    Q :: Vector{Matrix{Float64}}
    R :: Vector{Matrix{Float64}}
end

function get_scenario_cost(model :: Model, scen_tree :: ScenarioTree, cost :: Cost, node :: Int64)
    x = model[:x]
    u = model[:u]
    # No input for leaf nodes!
    res = x[node_to_x(scen_tree, node)]' * cost.Q[node_to_timestep(scen_tree, node)] * x[node_to_x(scen_tree, node)]

    while node != 1
        node = scen_tree.anc_mapping[node]
        res += (x[node_to_x(scen_tree, node)]' * cost.Q[node_to_timestep(scen_tree, node)] * x[node_to_x(scen_tree, node)] 
            + [u[node_to_timestep(scen_tree, node)]]' * cost.R[node_to_timestep(scen_tree, node)] * [u[node_to_timestep(scen_tree, node)]])
    end

    return res 
end

function construct_cost_matrix(scen_tree :: ScenarioTree, cost :: Cost, dynamics :: Dynamics)
    # # Inputs
    # Qs = [sparse(cost.Q[node_to_timestep(scen_tree, node)])]
    # node_T = node
    # while node != 1
    #     node = scen_tree.anc_mapping[node]
    #     append!(Qs, sparse(cost.Q[node_to_timestep(scen_tree, node)]))
    # end

    # # States


    # return 2 .* blockdiag(Qs...) # Compensate for factor 1 / 2

    scenarios = []
    for k = scen_tree.leaf_node_max_index:-1:scen_tree.leaf_node_min_index
        nn = k
        scenario = [nn]
        while nn != 1
            nn = scen_tree.anc_mapping[nn]
            append!(scenario, nn)
        end
        append!(scenarios, [reverse(scenario)])
    end

    # Now, scenarios contains a list of lists, where each list defines all nodes for a single scenario
    T = length(scen_tree.min_index_per_timestep)
    # Qs = []
    # qs = []
    cs = Matrix{Float64}[]
    # Rs = []
    for scen_ind = 1:length(scenarios)
        # Q = []
        # R = []
        # q = []
        Cs = Matrix{Float64}[]
        A_bars = Matrix{Float64}[]
        scenario = scenarios[scen_ind]
        for t = 1:T
            nn = scenario[t]
            w = scen_tree.node_info[nn].w
            if t == 1 # dynamics are stored on the node to which they lead
                continue
            elseif t == 2
                Cs = [dynamics.B[w]]
                A_bars = [dynamics.A[w]]
            else
                append!(Cs, [dynamics.A[w] * Cs[end]])
                append!(A_bars, [dynamics.A[w] * A_bars[end]])
            end
        end
        # println(Cs)
        # println(A_bars)
        # println("---")

        c = cost.Q[1]
        for t = 2:T
            c += A_bars[t - 1]' * cost.Q[t] * A_bars[t-1]
        end
        append!(cs, [c])
    end

end

function impose_cost(model :: Model, scen_tree :: ScenarioTree, cost :: Cost)
    # TODO: Could be more efficient by checking which indices have already been 
    # computed, but this function is called only once during the build step.
    @constraint(
        model,
        cost[i= scen_tree.leaf_node_min_index : scen_tree.leaf_node_max_index], # Only leaf nodes
        get_scenario_cost(model, scen_tree, cost, i) <= model[:s][scen_tree.node_info[i].s]
    )
end