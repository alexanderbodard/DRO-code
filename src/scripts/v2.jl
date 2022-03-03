
using ProximalOperators, Random, JuMP, MosekTools
import MathOptInterface as MOI
import LinearAlgebra as LA
Random.seed!(1234)

##########################
# Utilities
##########################

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
    - s: Conditional risk measure index in this node, given the current node. Always a scalar!

Not all of these variables are defined for all nodes of the scenario tree. In such case, nothing is returned.
The above variables are defined for:
    - x: all nodes
    - u: non-leaf nodes
    - w: non-root nodes
    - s: non-leaf nodes
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
    - N: Total number of nodes in this scenario tree
"""
struct ScenarioTree
    child_mapping :: Dict{Int64, Vector{Int64}}
    anc_mapping :: Dict{Int64, Int64}
    node_info :: Vector{ScenarioTreeNodeInfo}
    n_x :: Int64
    n_u :: Int64
    n :: Int64
    n_non_leaf_nodes :: Int64
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
    @constraint(
        model,
        dynamics[i=2:scen_tree.n], # Non-root nodes
        x[
            node_to_x(scen_tree, i)
        ] .== 
            dynamics.A[scen_tree.node_info[i].w] * node_to_x(scen_tree, scen_tree.anc_mapping[i]) 
            + dynamics.B[scen_tree.node_info[i].w] * node_to_u(scen_tree, scen_tree.anc_mapping[i])
    )
end

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

function get_all_scenario_paths()
# TODO
end

##########################
# Mosek reference implementation
##########################

###
# Problem statement
###

# Scenario tree
node_info = [
    ScenarioTreeNodeInfo(
        collect((i - 1) * 2 + 1 : i * 2),
        i < 4 ? [i] : nothing,
        i > 1 ? (i % 2) + 1 : nothing,
        i < 4 ? i : nothing
    ) for i in collect(1:7)
]

scen_tree = ScenarioTree(
    Dict(
        1 => [2, 3], 
        2 => [4, 5], 
        3 => [6, 7],
    ),
    Dict(
        2 => 1,
        3 => 1,
        4 => 2,
        5 => 2,
        6 => 3,
        7 => 3,
    ),
    node_info,
    2,
    1,
    7,
    3
)

# Dynamics: Based on a discretized car model
T_s = 0.05
n_x = 2
n_u = 1
d = 2
A = [[[1.,0.] [T_s, 1.0 - rand()*T_s]] for _ in 1:d]
B = [reshape([0., T_s], :, 1) for _ in 1:d]
dynamics = Dynamics(A, B, n_x, n_u)

###
# Formulate the optimization problem
###

# Define model, primal variables, epigraph variables and objective
model = Model(Mosek.Optimizer)
set_silent(model)

@variable(model, x[i=1:scen_tree.n * scen_tree.n_x])
@variable(model, u[i=1:scen_tree.n_non_leaf_nodes * scen_tree.n_u])
@variable(model, s[i=1:scen_tree.n_non_leaf_nodes * 1])

@objective(model, Min, s[1])

# Impose cost


# Impose dynamics
impose_dynamics(model, scen_tree, dynamics)

###
# Solve the optimization problem
###

optimize!(model)
println(value.(s))