
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
        dynamics[i=2:scen_tree.n], # Non-root nodes, so all except i = 1
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

function get_scenario_cost(scen_tree :: ScenarioTree, cost :: Cost, node :: Int64)
    # No input for leaf nodes!
    res = x[node_to_x(scen_tree, node)]' * cost.Q[node_to_timestep(scen_tree, node)] * x[node_to_x(scen_tree, node)]

    while node != 1
        node = scen_tree.anc_mapping[node]
        res += x[node_to_x(scen_tree, node)]' * cost.Q[node_to_timestep(scen_tree, node)] * x[node_to_x(scen_tree, node)] 
            + u[node_to_u(scen_tree, node)]' * cost.R[node_to_timestep(scen_tree, node)] * u[node_to_u(scen_tree, node)]
    end
    return res 
end

function impose_cost(model :: Model, scen_tree :: ScenarioTree, cost :: Cost)
    # Could be more efficient by checking which indices have already been 
    # computed, but this function is called only once during the build step.
    @constraint(
        model,
        cost[i= scen_tree.leaf_node_min_index : scen_tree.leaf_node_max_index], # Only leaf nodes
        get_scenario_cost(scen_tree, cost, i) <= s[scen_tree.node_info[i].s]
    )
end

###
# Risk constraints
###

struct ConvexCone
    subcones:: Array{MOI.AbstractVectorSet}
end

abstract type AbstractRiskMeasure end 

struct  Riskmeasure <: AbstractRiskMeasure
    A:: Matrix{Float64}
    B:: Matrix{Float64}
    b:: Vector{Float64}
    C:: ConvexCone
    D:: ConvexCone
end

# """
# x: primal vector (v, t) with v the argument, t the epigraph variable 
# y: dual vector 
# """ 
# function in_epigraph(r:: Riskmeasure, x_next::Vector, y::Vector)
#     return in(r.A' * x_next + r.B' * y , r.C.subcones[1])
# end

function add_risk_epi_constraint(model::Model, r::Riskmeasure, x_current, x_next::Vector)
    # TODO: Naming of the dual variables
    y = @variable(model, [1:length(r.b)])

    # 2b
    @constraint(model, in(r.A' * x_next + r.B' * y , r.C.subcones[1]))
    # 2c
    @constraint(model, in(y, r.D.subcones[1]))
    # 2d
    @constraint(model, - LA.dot(r.b, y) <= x_current)
end

function add_risk_epi_constraints(model::Model, scen_tree :: ScenarioTree, r::Vector{Riskmeasure})
    for i = 1:scen_tree.n_non_leaf_nodes
        add_risk_epi_constraint(
            model,
            r[i],
            s[i],
            s[scen_tree.child_mapping[i]]
        )
    end
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
        i,
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
    3,
    4,
    7,
    [1, 2, 4]
)

# Dynamics: Based on a discretized car model
T_s = 0.05
n_x = 2
n_u = 1
d = 2
A = [[[1.,0.] [T_s, 1.0 - rand()*T_s]] for _ in 1:d]
B = [reshape([0., T_s], :, 1) for _ in 1:d]
dynamics = Dynamics(A, B, n_x, n_u)

# Cost: Let's take a quadratic cost, equal at all timesteps
Q = LA.Matrix([2.2 0; 0 3.7])
R = reshape([3.2], 1, 1)
cost = Cost(
    collect([
        Q for _ in 1:3
    ]),
    collect([
        R for _ in 1:3
    ])
)

# Risk measures
rms = [
    Riskmeasure(
        LA.I(2),
        LA.I(2),
        [3,2],
        ConvexCone([MOI.Nonnegatives(2)]),
        ConvexCone([MOI.Nonnegatives(2)])
    ) for _ in 1:scen_tree.n_non_leaf_nodes
]

###
# Formulate the optimization problem
###

# Define model, primal variables, epigraph variables and objective
model = Model(Mosek.Optimizer)
set_silent(model)

@variable(model, x[i=1:scen_tree.n * scen_tree.n_x])
@variable(model, u[i=1:scen_tree.n_non_leaf_nodes * scen_tree.n_u])
@variable(model, s[i=1:scen_tree.n * 1])

@objective(model, Min, s[1])

# Impose cost
impose_cost(model, scen_tree, cost)

# Impose dynamics
impose_dynamics(model, scen_tree, dynamics)

# Impose risk measure epigraph constraints
add_risk_epi_constraints(model, scen_tree, rms)

###
# Solve the optimization problem
###

println(model)

optimize!(model)
println(value.(s))
println(value.(x))
println(value.(u))