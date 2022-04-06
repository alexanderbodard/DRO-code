###
# Dynamics
###

"""
Struct containing the dynamics in all relevant nodes. Indexing should be performed
with ScenarioTreeNodeInfo.w for a given node. Dynamics are assumed linear:
    x+ = A_i * x + B_i * u
    - A: Vector of matrices, index with w
    - B: Vector of matrices, index with w
    - n_x: Dimension of x in a single node
    - n_u: Dimension of u in a single node
TODO: Store n_x and n_u in both Dynamics and ScenarioTree, or only in of them?
"""
struct Dynamics
    A :: Vector{Matrix{Float64}}
    B :: Vector{Matrix{Float64}}
    n_x :: Int64
    n_u :: Int64
end

function impose_dynamics(model :: Model, scen_tree :: ScenarioTree, dynamics :: Dynamics)
    x = model[:x]
    u = model[:u]

    @constraint(
        model,
        dynamics[i=2:scen_tree.n], # Non-root nodes, so all except i = 1
        x[
            node_to_x(scen_tree, i)
        ] .== 
            dynamics.A[scen_tree.node_info[i].w] * x[node_to_x(scen_tree, scen_tree.anc_mapping[i])]
            + dynamics.B[scen_tree.node_info[i].w] * u[node_to_timestep(scen_tree, scen_tree.anc_mapping[i])]
    )
end