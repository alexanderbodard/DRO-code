
using ProximalOperators, Random, JuMP, MosekTools, SparseArrays
import MathOptInterface as MOI
import MathOptSetDistances as MOD
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
            dynamics.A[scen_tree.[i].w] * x[node_to_x(scen_tree, scen_tree.anc_mapping[i])]
            + dynamics.B[scen_tree.node_info[i].w] * u[node_to_u(scen_tree, scen_tree.anc_mapping[i])]
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

function add_risk_epi_constraint(model::Model, r::Riskmeasure, x_current, x_next::Vector, y)
    # 2b
    @constraint(model, in(-(r.A' * x_next + r.B' * y) , r.C.subcones[1]))
    # 2c
    @constraint(model, in(-y, r.D.subcones[1]))
    # 2d
    @constraint(model, - r.b' * y <= x_current)
end

function add_risk_epi_constraints(model::Model, scen_tree :: ScenarioTree, r::Vector{Riskmeasure})
    n_y = length(r[1].b)
    @variable(model, y[i=1:scen_tree.n_non_leaf_nodes * n_y])
    
    for i = 1:scen_tree.n_non_leaf_nodes
        add_risk_epi_constraint(
            model,
            r[i],
            model[:s][i],
            model[:s][scen_tree.child_mapping[i]],
            y[(i - 1) * n_y + 1 : n_y * i]
        )
    end
end

###
# Model
###

struct CustomModel{T, TT, TTT, U}
    L :: T
    Ltrans :: TT
    grad_f :: TTT
    prox_hstar_Sigmainv :: U
end

function primal_dual_alg(x, v, model :: CustomModel)
    n_y = length(rms[1].b)

    n_z = (scen_tree.n_non_leaf_nodes * scen_tree.n_u       # Every node has an input
                + n_x                                       # Only state at t=0
                + scen_tree.n                               # s variable: 1 component per node
                + scen_tree.n_non_leaf_nodes * n_y)         # One y variable for each non leaf node


    n_L_rows = (size(rms[1].A)[2]    # 4a
                + n_y               # 4b
                + 1 + n_y)           # 4c
    n_cost_i = (n_x                                     # x0
        + scen_tree.n_non_leaf_nodes * scen_tree.n_u    # u
        + 1)                                            # x_{T-1}[i]
    n_cost = n_cost_i * (scen_tree.n - scen_tree.n_non_leaf_nodes)
    n_L = scen_tree.n_non_leaf_nodes * n_L_rows + n_cost

    # TODO: Initialize Sigma and Gamma in some way
    """
    - Sigma in S_{++}^{n_L}
    - Gamma in S_{++}^{n_z}
    """
    Sigma = sparse(LA.Diagonal(1:1:n_L))
    Gamma = sparse(LA.Diagonal(1:1:n_z))
    lambda = 1

    # TODO: Work with some tolerance
    counter = 0
    while counter < 2
        xbar = x - Gamma * model.Ltrans(v) - Gamma * model.grad_f(x)
        vbar = model.prox_hstar_Sigmainv(v + Sigma * model.L(2 * xbar - x), 1 ./ Sigma)
        x = lambda * xbar + (1 - lambda) * x
        v = lambda * vbar + (1 - lambda) * v

        counter += 1
    end
end

function build_custom_model(scen_tree :: ScenarioTree, cost :: Cost, rms :: Vector{Riskmeasure}, x0, u0, s0, y0)
    n_y = length(rms[1].b)

    n_z = (scen_tree.n_non_leaf_nodes * scen_tree.n_u        # Every node has an input
                + n_x                                       # Only state at t=0
                + scen_tree.n                               # s variable: 1 component per node
                + scen_tree.n_non_leaf_nodes * n_y)         # One y variable for each non leaf node
    
    """
    We will structure the vector z as follows:
    - z[
        1:n_x
    ] = x_0
    - z[
        n_x + 1 : 
        n_x + scen_tree.n_non_leaf_nodes * scen_tree.n_u
    ] = u
    - z[
        n_x + scen_tree.n_non_leaf_nodes * scen_tree.n_u + 1 : 
        n_x + scen_tree.n_non_leaf_nodes * scen_tree.n_u + scen_tree.n
    ] = s
    - z[
        n_x + scen_tree.n_non_leaf_nodes * scen_tree.n_u + scen_tree.n + 1 : 
        n_x + scen_tree.n_non_leaf_nodes * scen_tree.n_u + scen_tree.n + scen_tree.n_non_leaf_nodes * n_y
    ] = y
    """

    z = repeat([NaN], n_z)

    z[z_to_x(scen_tree)] = x0
    z[z_to_u(scen_tree)] = u0
    z[z_to_s(scen_tree)] = s0
    z[z_to_y(scen_tree, n_y)] = y0

    n_L_rows = (size(rms[1].A)[2]    # 4a
                + n_y               # 4b
                + 1 + n_y)           # 4c
    n_cost_i = (n_x                                     # x0
        + scen_tree.n_non_leaf_nodes * scen_tree.n_u    # u
        + 1)                                            # x_{T-1}[i]
    n_cost = n_cost_i * (scen_tree.n - scen_tree.n_non_leaf_nodes)
    n_L = scen_tree.n_non_leaf_nodes * n_L_rows + n_cost

    # 4a
    """

    """
    L_I = Float64[]
    L_J = Float64[]
    L_V = Float64[]

    n_y_start_index = (scen_tree.n_non_leaf_nodes * scen_tree.n_u  # Inputs
                        + n_x                                     # State at t=0
                        + scen_tree.n                             # S variables
                        + 1)

    ss = z_to_s(scen_tree)
    yy = z_to_y(scen_tree, n_y)
    for k = 1:scen_tree.n_non_leaf_nodes
        Is = Float64[]
        Js = Float64[]
        Vs = Float64[]

        Js = ss[scen_tree.child_mapping[k]]
        append!(Is, [i for i in collect(1:length(Js))])
        append!(Vs, [1 for _ in 1:length(Js)])
        S_s = sparse(Is, Js, Vs, length(Is), n_y_start_index - 1)

        Is = Float64[]
        Js = Float64[]
        Vs = Float64[]

        Js = yy[(k - 1) * n_y + 1 : k * n_y] .- (n_y_start_index - 1)
        append!(Is, [i for i in collect(1:length(Js))])
        append!(Vs, [1 for _ in 1:length(Js)])
        S_y = sparse(Is, Js, Vs, length(Is), n_z - n_y_start_index + 1)

        r = rms[k]
        S = hcat(sparse(r.A') * S_s, sparse(r.B') * S_y)
        SI, SJ, SV = findnz(S)
        if k > 1
            append!(L_I, SI .+ maximum(L_I))
        else
            append!(L_I, SI)
        end
        append!(L_J, SJ)
        append!(L_V, SV)
    end

    # 4b
    """
        L_I: Just one element per row => 1, 2, 3...
        L_J: z_to_y
        L_V: all ones
    """
    L_II = Float64[]
    L_JJ = Float64[]
    L_VV = Float64[]

    yy = z_to_y(scen_tree, n_y)
    for k = 1:scen_tree.n_non_leaf_nodes
        ind = collect(
        (k - 1) * n_y + 1 : k * n_y
        )
        append!(L_JJ, yy[ind])
    end
    append!(L_II, [i for i in collect(1 : scen_tree.n_non_leaf_nodes * n_y)])
    append!(L_VV, [1 for _ in 1:scen_tree.n_non_leaf_nodes * n_y])
    
    # 4c
    """
        L_I: Just one element per row => 1, 2, 3...
        L_J: z_to_s, z_to_y
        L_V: all negative ones
    """
    L_III = Float64[]
    L_JJJ = Float64[]
    L_VVV = Float64[]

    yy = z_to_y(scen_tree, n_y)
    ss = z_to_s(scen_tree)
    for k = 1:scen_tree.n_non_leaf_nodes
        append!(L_JJJ, ss[k])
        ind = collect(
        (k - 1) * n_y + 1 : k * n_y
        )
        append!(L_JJ, yy[ind])
    end
    append!(L_III, [i for i in collect(1 : scen_tree.n_non_leaf_nodes * (n_y + 1))])
    append!(L_VVV, [-1 for _ in 1:scen_tree.n_non_leaf_nodes * (n_y + 1)])    

    # 4d
    L_IIII = Float64[]
    L_JJJJ = Float64[]
    L_VVVV = Float64[]

    xx = z_to_x(scen_tree)
    uu = z_to_u(scen_tree)
    for k = scen_tree.leaf_node_min_index : scen_tree.leaf_node_max_index
        append!(L_JJJJ, xx)
        append!(L_JJJJ, uu)
        append!(L_JJJJ, ss[k])
    end
    append!(L_IIII, [i for i in 1 : length(L_JJJJ)])
    append!(L_VVVV, [1 for _ in 1 : length(L_JJJJ)])


    append!(L_I, L_II .+ maximum(L_I))
    append!(L_I, L_III .+ maximum(L_I))
    append!(L_I, L_IIII .+ maximum(L_I))
    append!(L_J, L_JJ, L_JJJ, L_JJJJ)
    append!(L_V, L_VV, L_VVV, L_VVVV)
    L = sparse(L_I, L_J, L_V, n_L, n_z)

    model = CustomModel(
        z -> L*z,
        z -> L'*z,
        x -> (
            temp = zeros(length(x));
            temp[1] = 1;
            temp
        ),
        (z, gamma) -> begin
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
            end

            # 4d
            for k = scen_tree.leaf_node_min_index : scen_tree.leaf_node_max_index
                break # TODO
            end

            # z - gamma proj(x ./ gamma, )
            z # TODO
        end
    )

    return model
end

function solve_custom_model(model :: CustomModel)
    n_y = length(rms[1].b)

    n_z = (scen_tree.n_non_leaf_nodes * scen_tree.n_u       # Every node has an input
                + n_x                                       # Only state at t=0
                + scen_tree.n                               # s variable: 1 component per node
                + scen_tree.n_non_leaf_nodes * n_y)         # One y variable for each non leaf node


    n_L_rows = (size(rms[1].A)[2]    # 4a
                + n_y               # 4b
                + 1 + n_y)           # 4c
    n_cost_i = (n_x                                     # x0
        + scen_tree.n_non_leaf_nodes * scen_tree.n_u    # u
        + 1)                                            # x_{T-1}[i]
    n_cost = n_cost_i * (scen_tree.n - scen_tree.n_non_leaf_nodes)
    n_L = scen_tree.n_non_leaf_nodes * n_L_rows + n_cost

    z = zeros(n_z)
    v = zeros(n_L)

    primal_dual_alg(z, v, model)
end

@enum Solver begin
    MOSEK_SOLVER = 1
    CUSTOM_SOLVER = 2
end

function build_model(scen_tree :: ScenarioTree, solver :: Solver)
    if solver == MOSEK_SOLVER
        # Define model, primal variables, epigraph variables and objective
        model = Model(Mosek.Optimizer)
        set_silent(model)

        @variable(model, x[i=1:scen_tree.n * scen_tree.n_x])
        @variable(model, u[i=1:scen_tree.n_non_leaf_nodes * scen_tree.n_u])
        @variable(model, s[i=1:scen_tree.n * 1])

        @objective(model, Min, s[1])

        # TODO: Initial condition
        @constraint(model, intial_condition[i=1:2], x[i] .== [2., 2.][i])

        # Impose cost
        impose_cost(model, scen_tree, cost)

        # Impose dynamics
        impose_dynamics(model, scen_tree, dynamics)

        # Impose risk measure epigraph constraints
        add_risk_epi_constraints(model, scen_tree, rms)

        return model
    end
    if solver == CUSTOM_SOLVER
        x0 = [2., 2.]
        u0 = zeros(scen_tree.n_non_leaf_nodes * scen_tree.n_u)
        s0 = zeros(scen_tree.n)
        y0 = zeros(scen_tree.n_non_leaf_nodes * length(rms[1].b))
        model = build_custom_model(scen_tree, cost, rms, x0, u0, s0, y0)
        return model
    end
end

function solve_model(model :: Union{Model, CustomModel}, solver :: Solver)
    if solver == MOSEK_SOLVER
        optimize!(model)
        println(solution_summary(model, verbose=true))
        return nothing
    end
    if solver == CUSTOM_SOLVER
        solve_custom_model(model)
        return nothing
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

# Risk measures: Risk neutral: A = I, B = [I; -I], b = [1;1;-1;-1]
"""
Risk neutral: A = I, B = [I; -I], b = [0.5;0.5;-0.5;-0.5]
AVaR: A = I, B = [-I, I, 1^T, -1^T], b = [0; p / alpha; 1, -1]
"""
rms = [
    Riskmeasure(
        LA.I(2),
        [LA.I(2); -LA.I(2)],
        [0.5 , 0.5, -0.5, -0.5],
        ConvexCone([MOI.Nonnegatives(2)]),
        ConvexCone([MOI.Nonnegatives(4)])
    ) for _ in 1:scen_tree.n_non_leaf_nodes
]

###
# Formulate the optimization problem
###

model = build_model(scen_tree, CUSTOM_SOLVER)

###
# Solve the optimization problem
###

solve_model(model, CUSTOM_SOLVER)