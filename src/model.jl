include("risk_constraints.jl")
include("custom_model.jl")
include("model_h_x.jl")

###
# Model
###

# struct CustomModel{T, TT, TTT, U}
#     L :: T
#     Ltrans :: TT
#     grad_f :: TTT
#     prox_hstar_Sigmainv :: U
# end

function primal_dual_alg(x, v, model :: CustomModel)
    n_z = length(x)
    n_L = length(v)

    # TODO: Initialize Sigma and Gamma in some way
    """
    - Sigma in S_{++}^{n_L}
    - Gamma in S_{++}^{n_z}
    """
    sigma = rand() * 1e-3
    gamma = rand() * 1e-3
    Sigma = rand() * sparse(LA.I(n_L))
    Gamma = rand() * sparse(LA.I(n_z))
    lambda = 1

    # TODO: Work with some tolerance
    counter = 0
    while counter < 5
        xbar = x - Gamma * model.Ltrans(v) - Gamma * model.grad_f(x)
        vbar = model.prox_hstar_Sigmainv(v + Sigma * model.L(2 * xbar - x), 1 ./ sigma)
        x = lambda * xbar + (1 - lambda) * x
        v = lambda * vbar + (1 - lambda) * v

        println(x)

        counter += 1
    end
end

# function get_n_z(scen_tree :: ScenarioTree, rms :: Vector{Riskmeasure}, eliminate_states :: Bool)
#     if eliminate_states
#         n_x = 0                             # Eliminate state variables
#     else
#         n_x = scen_tree.n * scen_tree.n_x   # Every node has a state
#     end

#     return (scen_tree.n_non_leaf_nodes * scen_tree.n_u              # Every non leaf node has an input
#                 + n_x                                               # n_x
#                 + scen_tree.n                                       # s variable: 1 component per node
#                 + scen_tree.n_non_leaf_nodes * length(rms[1].b))    # One y variable for each non leaf node
# end

# function get_n_L(scen_tree :: ScenarioTree, rms :: Vector{Riskmeasure}, eliminate_states :: Bool)
#     n_y = length(rms[1].b)
#     if eliminate_states
#         n_x = 0                             # Eliminate state variables
#     else
#         n_x = scen_tree.n * scen_tree.n_x   # Every node has a state
#     end

#     n_L_rows = (size(rms[1].A)[2]    # 4a
#                 + n_y                # 4b
#                 + 1 + n_y)           # 4c
#     n_cost_i = (n_x                                     # x0
#         + scen_tree.n_non_leaf_nodes * scen_tree.n_u    # u
#         + 1)                                            # x_{T-1}[i]
#     n_cost = n_cost_i * (scen_tree.n - scen_tree.n_non_leaf_nodes)
#     return scen_tree.n_non_leaf_nodes * n_L_rows + n_cost
# end

function build_custom_model(scen_tree :: ScenarioTree, cost :: Cost, rms :: Vector{Riskmeasure}, z0, x0)
    eliminate_states = true
    # n_z = get_n_z(scen_tree, rms, eliminate_states)
    n_z = length(z0)
    z = z0

    n_L = get_n_L(scen_tree, rms, eliminate_states)

    # TODO: Remove from here
    construct_cost_matrix(scen_tree, cost, dynamics)

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
    n_z = get_n_z(scen_tree, rms, false)
    n_L = get_n_L(scen_tree, rms, false)

    z = zeros(n_z)
    v = zeros(n_L)

    primal_dual_alg(z, v, model)
end

@enum Solver begin
    MOSEK_SOLVER = 1
    CUSTOM_SOLVER = 2
    H_X_SOLVER = 3
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
        z0 = vcat(u0, s0, y0)
        model = build_custom_model(scen_tree, cost, rms, z0, x0)
        return model
    end
    if solver == H_X_SOLVER
        return build_h_x_model(scen_tree, rms)     
    end
end

function solve_model(model :: Union{Model, CustomModel}, solver :: Solver)
    if solver == MOSEK_SOLVER
        optimize!(model)
        # println(solution_summary(model, verbose=true))
        return value.(model[:x]), value.(model[:u])
    end
    if solver == CUSTOM_SOLVER
        solve_custom_model(model)
        return nothing, nothing
    end
    if solver == H_X_SOLVER
        solve_custom_model(model)
        return nothing, nothing
    end
end