###
# Scenario tree
###

"""
Struct that stores all relevant information for some node of the scenario tree.
    - x: Indices of state variables belonging to this node. In general, this is vector valued.
    - u: Indices of input variables belonging to this node. In general, this is vector valued.
    - w: Represents the uncertainty in dynamics when going from the parent of the given
        node to the given node. This integer is used as an index to retrieve these dynamics
        from the Dynamics struct.
    - s: Non-leaf nodes: Conditional risk measure index in this node, given the current node. 
         Leaf nodes: Index of the total cost of this scenario
        Always a scalar!

Not all of these variables are defined for all nodes of the scenario tree. In such case, nothing is returned.
The above variables are defined for:
    - x: all nodes
    - u: non-leaf nodes
    - w: non-root nodes
    - s: non-leaf nodes for risk measures values, leaf nodes for cost values of corresponding scenario
"""
struct ScenarioTreeNodeInfo
    x :: Union{Vector{Int64}, Nothing}
    u :: Union{Vector{Int64}, Nothing}
    w :: Union{Int64, Nothing}
    s :: Union{Int64, Nothing}
end

"""
Struct that represents a scenario tree.
    - child_mapping: Dictionary that maps node indices to a vector of child indices
    - anc_mapping: Dictionary that maps node indices to their parent node indices
    - node_info: All relevant information, indexable by the node index
    - n_x: Number of components of a state vector in a single node
    - n_u: Number of components of an input vector in a single node
    - n: Total number of nodes in this scenario tree
"""
struct ScenarioTree
    child_mapping :: Dict{Int64, Vector{Int64}}
    anc_mapping :: Dict{Int64, Int64}
    node_info :: Vector{ScenarioTreeNodeInfo}
    n_x :: Int64
    n_u :: Int64
    n :: Int64
    n_non_leaf_nodes :: Int64
    leaf_node_min_index :: Int64
    leaf_node_max_index :: Int64
    min_index_per_timestep :: Vector{Int64}
end

function node_to_x(scen_tree :: ScenarioTree, i :: Int64)
    return collect(
        (i - 1) * scen_tree.n_x + 1 : i * scen_tree.n_x
    )
end

function node_to_u(scen_tree :: ScenarioTree, i :: Int64)
    return collect(
        (i - 1) * scen_tree.n_u + 1 : i * scen_tree.n_u
    )
end

function node_to_timestep(scen_tree :: ScenarioTree, i :: Int64)
    for j = 1:length(scen_tree.min_index_per_timestep)
        if (i < scen_tree.min_index_per_timestep[j])
            return j - 1
        end
    end
    return length(scen_tree.min_index_per_timestep)
end

"""
Get the indices of the x variable's components. 
Note that in this formulation only x0 is used, so no index for x must be provided as in the other similar functions. 
"""
function z_to_x(scen_tree :: ScenarioTree)
    return collect(
        1 : scen_tree.n_x
    )
end    

function z_to_u(scen_tree :: ScenarioTree)
    return collect(
        scen_tree.n_x + 1 : 
        scen_tree.n_x + scen_tree.n_non_leaf_nodes * scen_tree.n_u
    )
end

function z_to_s(scen_tree :: ScenarioTree)
    return collect(
        scen_tree.n_x + scen_tree.n_non_leaf_nodes * scen_tree.n_u + 1 :
        scen_tree.n_x + scen_tree.n_non_leaf_nodes * scen_tree.n_u + scen_tree.n
    )
end

function z_to_y(scen_tree :: ScenarioTree, n_y :: Int64)
    return collect(
        scen_tree.n_x + scen_tree.n_non_leaf_nodes * scen_tree.n_u + scen_tree.n + 1 :
        scen_tree.n_x + scen_tree.n_non_leaf_nodes * scen_tree.n_u + scen_tree.n + scen_tree.n_non_leaf_nodes * n_y
    )
end