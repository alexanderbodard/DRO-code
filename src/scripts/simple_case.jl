
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

struct ScenarioTree
    n_x :: Int64
    smallest_leaf_node_ind :: Int64
    child_mapping :: Dict{Int64, Vector{Int64}}
    anc_mapping :: Dict{Int64, Int64}
end

"""
tree: Scenario tree
p_ind: Parent node index for which the children are wanted
x: Vector to be indexed with the child nodes

Note: This function assumes that the given node is no leaf node.
NaN is returned in this case.
"""
function get_child_values(tree::ScenarioTree, p_ind :: Int64, x::Vector{VariableRef})
    if p_ind < tree.smallest_leaf_node_ind
        # Non-leaf nodes
        c_inds = tree.child_mapping[p_ind]
        inds = Array{Int64}(undef, length(c_inds) * tree.n_x)
        for (i, c_ind) in enumerate(c_inds)
            inds[
                [
                    j for j = (i - 1) * tree.n_x + 1 : i * tree.n_x
                ]
            ] = [
                    j for j = (c_ind - 1) * tree.n_x + 1 : c_ind * tree.n_x
                ]
        end
        return x[inds]
    end
    # Return NaN in case of leaf nodes
    return [NaN for _ in 1:tree.n_x]
end

function get_ancestor_ind(tree::ScenarioTree, p_ind :: Int64)
    if p_ind > 1
        return tree.anc_mapping[p_ind]
    end
    # Return NaN in case of root node
    return NaN
end

# scen_tree = ScenarioTree(3, 4, Dict(1 => [2, 3], 2 => [4, 5], 3 => [6, 7]))
# z = [i * 1. for i in 1:21]
# res = get_child_values(scen_tree, 1, z)

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

"""
x: primal vector (v, t) with v the argument, t the epigraph variable 
y: dual vector 
""" 
function in_epigraph(r:: Riskmeasure, x::Vector, y::Vector)
    v, t = x[1:end-1], x[end]    

    return in(r.A' * v + r.B' * y , r.C.subcones[1]) 
end

function add_risk_epi_constraint(model::Model, r::Riskmeasure, x::Vector)
    # TODO: Naming of the dual variables
    y = @variable(model, [1:length(r.b)])
    @constraint(model, y .>= 0.)

    # 2c
    @constraint(model, in(y, r.D.subcones[1]))
    # 2d
    @constraint(model, - LA.dot(r.b, y) .<= x)
end

r = Riskmeasure(LA.I(3), LA.I(3), [3,2,2], ConvexCone([MOI.Nonnegatives(3)]), ConvexCone([MOI.Nonnegatives(3)]))
# add_risk_epi_constraint(model, r)

###
# Dynamics
###

struct Dynamics
    A :: Vector{Matrix{Float64}}
    B :: Vector{Matrix{Float64}}
    n_x :: Int64
    n_u :: Int64
end

###
# Helper function with indexing
###
function get_z_from_node_index_start(i:: Int64, n_node :: Int64)
    return (i - 1) * n_node + 1
end

function get_z_from_node_index_end(i:: Int64, n_node :: Int64)
    return i * n_node
end

##########################
# Mosek reference implementation
##########################

###
# Problem statement
###

# Scenario tree: at a given time, branching factor is assumed equal for all nodes
scen_tree = ScenarioTree(3, 
    4, 
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
    ))
scen_total_nodes = 7
scen_nodes_per_timestep = [1, 2, 4]
n_node = 3
d = 2
T = 3

# Dynamics: Based on a discretized car model
T_s = 0.05
n_x = 2
n_u = 1
A = [[[1.,0.] [T_s, 1.0 - rand()*T_s]] for _ in 1:sum(scen_nodes_per_timestep[2:end])]
B = [reshape([0., T_s], :, 1) for _ in 1:sum(scen_nodes_per_timestep[2:end])]
dynamics = Dynamics(A, B, n_x, n_u)

# Cost function: Quadratic
Q = LA.Diagonal([2.2, 3.7])
R = LA.Diagonal([3.2])
QR = LA.Diagonal([2.2, 3.7, 3.2])
QQRR = LA.Diagonal([2.2, 3.7, 3.2] * scen_total_nodes)

# Lower / Upper bounds on the optimization variables
# lb_x = repeat([0. 0.], d)
# ub_x = repeat([100. 10.], d)
# lb_y = repeat([0.], d)
# ub_y = repeat([2.674], d)

# Risk measures
risk_measures = Dict(
    1 => Riskmeasure(LA.I(3) .+ rand(), LA.I(3) .+ rand(), [3,2,2], ConvexCone([MOI.Nonnegatives(3)]), ConvexCone([MOI.Nonnegatives(3)])),
    2 => Riskmeasure(LA.I(3) .+ rand(), LA.I(3) .+ rand(), [3,2,2], ConvexCone([MOI.Nonnegatives(3)]), ConvexCone([MOI.Nonnegatives(3)])),
    3 => Riskmeasure(LA.I(3) .+ rand(), LA.I(3) .+ rand(), [3,2,2], ConvexCone([MOI.Nonnegatives(3)]), ConvexCone([MOI.Nonnegatives(3)])),
    4 => Riskmeasure(LA.I(3) .+ rand(), LA.I(3) .+ rand(), [3,2,2], ConvexCone([MOI.Nonnegatives(3)]), ConvexCone([MOI.Nonnegatives(3)])),
    5 => Riskmeasure(LA.I(3) .+ rand(), LA.I(3) .+ rand(), [3,2,2], ConvexCone([MOI.Nonnegatives(3)]), ConvexCone([MOI.Nonnegatives(3)])),
    6 => Riskmeasure(LA.I(3) .+ rand(), LA.I(3) .+ rand(), [3,2,2], ConvexCone([MOI.Nonnegatives(3)]), ConvexCone([MOI.Nonnegatives(3)])),
    7 => Riskmeasure(LA.I(3) .+ rand(), LA.I(3) .+ rand(), [3,2,2], ConvexCone([MOI.Nonnegatives(3)]), ConvexCone([MOI.Nonnegatives(3)])),
)

###
# Formulate the optimization problem
###

# Define model, primal variables and objective
model = Model(Mosek.Optimizer)
set_silent(model)

n_z = (7 + 2 + 1) * n_node
@variable(model, z[i=1:n_z])

@objective(model, Min, z[1])

# Impose the cost
z_T = z[1+sum(scen_nodes_per_timestep[1:end-1]) * n_node : end]
z_TT = z[1+ sum(scen_nodes_per_timestep[1:end-2]) * n_node: sum(scen_nodes_per_timestep[1:end-1]) * n_node]
@constraint(
    model, 
    cost[i = 1:scen_nodes_per_timestep[end]], 
    z_T[
        (i - 1) * n_node + 1 : i * n_node
    ]' * QR * z_T[
        (i - 1) * n_node + 1 : i * n_node
    ] .<= z_TT[
        i ÷ (d * n_node + 1) * n_node + (i % 3) + 1
    ]
)

# Impose epigraph constraints
add_risk_epi_constraint(model, r, z[1:3])

# Impose dynamics
@constraint(
    model,
    dynamics[i = 2:sum(scen_nodes_per_timestep[2:end])],
    z_T[
        get_z_from_node_index_start(i, n_node) : get_z_from_node_index_end(i, n_node) - n_u
    ] .== dynamics.A[i] * z_T[
        get_z_from_node_index_start(
            get_ancestor_ind(scen_tree, i), n_node
        ) : get_z_from_node_index_end(
            get_ancestor_ind(scen_tree, i), n_node
        ) - n_u
    ] + dynamics.B[i] * z_T[
        get_z_from_node_index_start(
            get_ancestor_ind(scen_tree, i), n_node
        ) + n_x : get_z_from_node_index_end(
            get_ancestor_ind(scen_tree, i), n_node
        )
    ]
)

###
# Solve the optimization problem
###

optimize!(model)
println(value.(z))