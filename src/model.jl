include("risk_constraints.jl")

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
        # println(solution_summary(model, verbose=true))
        return value.(model[:x]), value.(model[:u])
    end
    if solver == CUSTOM_SOLVER
        solve_custom_model(model)
        return nothing
    end
end