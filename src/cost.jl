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
        res += x[node_to_x(scen_tree, node)]' * cost.Q[node_to_timestep(scen_tree, node)] * x[node_to_x(scen_tree, node)] 
            + u[node_to_u(scen_tree, node)]' * cost.R[node_to_timestep(scen_tree, node)] * u[node_to_u(scen_tree, node)]
    end
    return res 
end

# function construct_cost_matrix(scen_tree :: ScenarioTree, cost :: Cost, node :: Int64)
#     # Inputs
#     Qs = [sparse(cost.Q[node_to_timestep(scen_tree, node)])]
#     node_T = node
#     while node != 1
#         node = scen_tree.anc_mapping[node]
#         append!(Qs, sparse(cost.Q[node_to_timestep(scen_tree, node)]))
#     end

#     # States


#     return 2 .* blockdiag(Qs...) # Compensate for factor 1 / 2
# end

function impose_cost(model :: Model, scen_tree :: ScenarioTree, cost :: Cost)
    # TODO: Could be more efficient by checking which indices have already been 
    # computed, but this function is called only once during the build step.
    @constraint(
        model,
        cost[i= scen_tree.leaf_node_min_index : scen_tree.leaf_node_max_index], # Only leaf nodes
        get_scenario_cost(model, scen_tree, cost, i) <= model[:s][scen_tree.node_info[i].s]
    )
end