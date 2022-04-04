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
        1 : scen_tree.n_x * scen_tree.n
    )
end    

function z_to_u(scen_tree :: ScenarioTree)
    return collect(
        scen_tree.n_x * scen_tree.n + 1 : 
        scen_tree.n_x * scen_tree.n + scen_tree.n_non_leaf_nodes * scen_tree.n_u
    )
end

function z_to_s(scen_tree :: ScenarioTree)
    return collect(
        scen_tree.n_x * scen_tree.n + scen_tree.n_non_leaf_nodes * scen_tree.n_u + 1 :
        scen_tree.n_x * scen_tree.n + scen_tree.n_non_leaf_nodes * scen_tree.n_u + scen_tree.n
    )
end

function z_to_y(scen_tree :: ScenarioTree, n_y :: Int64)
    return collect(
        scen_tree.n_x * scen_tree.n + scen_tree.n_non_leaf_nodes * scen_tree.n_u + scen_tree.n + 1 :
        scen_tree.n_x * scen_tree.n + scen_tree.n_non_leaf_nodes * scen_tree.n_u + scen_tree.n + scen_tree.n_non_leaf_nodes * n_y
    )
end

function plot_scen_tree_x_i(scen_tree :: ScenarioTree, x :: Vector{Float64}, i :: Int64, filename :: String)
    pgfplotsx()

    xs = [1]
    ys = [0.]

    parents = [1]
    parent_height = Dict(1 => 2.)
    parent_offset = Dict(1 => 1.)
    while length(parents) > 0
        parent = parents[1]
        parents = parents[2:end]
        children = scen_tree.child_mapping[parent]
        println(parent, children)
        for child in children
            append!(xs, [node_to_timestep(scen_tree, child)])

            parent_height[child] = parent_height[parent] / length(children)
            parent_offset[child] = parent_offset[parent] - parent_height[child] * (child - minimum(children))
            append!(ys, parent_offset[child] - parent_height[child] / 2 )
        end
        for child in children
            if child < scen_tree.leaf_node_min_index
                append!(parents, child)
            end
        end
    end
    
    println(xs, ys)
    scatter(xs, ys, fmt=:png, xlim = (0.6, length(scen_tree.min_index_per_timestep) + 2.2), ylim = (-1.1, 1.1), marker = (10, 0.2, :orange), series_annotations = text.(1:7), label="")
    annotate!(collect(zip(xs .+ 0.1, ys, map(data -> (data[i], 16, :left), collect(eachrow(reshape(x, scen_tree.n_x, 7)'))))))
    filename = string(filename, ".png")
    savefig(filename)
end

function plot_scen_tree_x(scen_tree :: ScenarioTree, x :: Vector{Float64}, filename :: String)
    for i = 1:scen_tree.n_x
        plot_scen_tree_x_i(scen_tree, x, i, string(filename, "_", i))
    end
end

function generate_scenario_tree(N :: Int64, d :: Int64, nx :: Int64, nu :: Int64)
    if d <= 1
        error("Branching factor d must be larger than 1, but is $(d).")
    end
    
    # Total number of nodes in the scenario tree
    n_total = (d^N - 1) / (d - 1)
    # Total number of leaf nodes
    n_leafs = d^(N - 1)
    # Total number of non-leaf nodes
    n_non_leafs = (d^(N - 1) - 1) / (d - 1)

    node_info = [
        ScenarioTreeNodeInfo(
            collect((i - 1) * d + 1 : i * d),
            i <= n_non_leafs ? [i] : nothing,
            i > 1 ? (i % d) + 1 : nothing,
            i,
        ) for i in collect(1:n_total)
    ]

    child_mapping = Dict()
    child_index = 2
    for i = 1:n_non_leafs
        child_mapping[i] = collect(child_index : child_index + d - 1)
        child_index += d
    end

    anc_mapping = Dict()
    for (key, value) in child_mapping
        for v in value
            anc_mapping[v] = key
        end
    end


    return ScenarioTree(
        child_mapping,
        anc_mapping,
        node_info,
        nx,
        nu,
        n_total,
        n_non_leafs,
        n_non_leafs + 1,
        n_total,
        vcat([1], [1 + (d^(i) - 1)/(d-1) for i in collect(1:N-1)])
    )
end