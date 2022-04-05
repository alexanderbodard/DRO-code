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
    # Currently they are defined as c * LA.I, so that the bisection
    # method can be used to retrieve c
    """
    - Sigma in S_{++}^{n_L}
    - Gamma in S_{++}^{n_z}

    Choose sigma and gamma such that
    sigma * gamma * model.L_norm < 1
    """
    sigma = 1e-1
    gamma = 1e-1
    sigma = 0.4
    gamma = 0.4
    Sigma = sigma * sparse(LA.I(n_L))
    Gamma = gamma * sparse(LA.I(n_z))
    lambda = 0.8

    if (sigma * gamma * model.L_norm > 1)
        error("sigma and gamma are not chosen correctly")
    end

    plot_vector = zeros(5000, 11)
    x0 = x[1:3]

    # TODO: Work with some tolerance
    counter = 0
    while counter < 5000
        x_old = copy(x) # TODO: Check Julia behaviour

        xbar = x - Gamma * model.Ltrans(v) - Gamma * model.grad_f(x)
        vbar = model.prox_hstar_Sigmainv(v + Sigma * model.L(2 * xbar - x), 1 ./ sigma, counter % 100 == 0)

        if counter == 1
            println(vbar)
        end

        x = lambda * xbar + (1 - lambda) * x
        v = lambda * vbar + (1 - lambda) * v

        plot_vector[counter + 1, 1:end] = x

        # if counter < 3
        #     println("-----------")
        #     println("x: ", x[z_to_x(scen_tree)])
        #     println("u: ", x[z_to_u(scen_tree)])
        #     println("s: ", x[z_to_s(scen_tree)])
        #     println("y: ", x[z_to_y(scen_tree, 4)])
        #     println("v: ", v)
        # end

        # println("Solution norm: ", LA.norm((x - x_old) / x) / length(x))
        if LA.norm((x - x_old) / x) / length(x) < 1e-12
            println("Breaking!", counter)
            break
        end
        counter += 1
    end

    residues = Float64[]
    for i = 1:counter
        append!(residues, LA.norm(plot_vector[i, 1:3] .- x_ref') / LA.norm(x0 .- x_ref .+ 1e-15))
    end

    # println(size(plot_vector))
    pgfplotsx()
    plot(residues, fmt = :png, labels=["x_residue"], xaxis=:log, yaxis=:log)
    # plot!(plot_vector[1:counter, 1:3], fmt = :png, labels=["x"])
    filename = "debug_x_res.png"
    savefig(filename)

    plot(plot_vector[1:counter, 1:3], fmt = :png, labels=["x"])
    # plot!(plot_vector[1:counter, 1:3], fmt = :png, labels=["x"])
    filename = "debug_x.png"
    savefig(filename)

    plot(plot_vector[1:counter, 4], fmt = :png, labels=["u"])
    filename = "debug_u.png"
    savefig(filename)

    plot(plot_vector[1:counter, 5:7], fmt = :png, labels=["s"])
    filename = "debug_s.png"
    savefig(filename)

    plot(plot_vector[1:counter, 8:11], fmt = :png, labels=["y"])
    filename = "debug_y.png"
    savefig(filename)

    return x
end

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
        end,
        nothing
    )

    return model
end

function solve_custom_model(model :: CustomModel)
    n_z = get_n_z(scen_tree, rms, false)
    n_L = get_n_L(scen_tree, rms, false)

    z = zeros(n_z)
    # z[1:3] = x_ref
    # z[4] = u_ref[1]
    # z[5:7] = s_ref
    # z[8:11] = y_ref
    v = zeros(n_L)

    z = primal_dual_alg(z, v, model)

    println("x: ", z[
        1:scen_tree.n * scen_tree.n_x
    ])

    println("u: ", z[z_to_u(scen_tree)])

    println("s: ", z[z_to_s(scen_tree)])

    println("y: ", z[z_to_y(scen_tree, 4)])

    return z[
        1:scen_tree.n * scen_tree.n_x
    ], z[z_to_u(scen_tree)]    
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
        @constraint(model, intial_condition[i=1:1], x[i] .== [2.][i])

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
        return value.(model[:x]), value.(model[:u]), value.(model[:s]), value.(model[:y])
    end
    if solver == CUSTOM_SOLVER
        solve_custom_model(model)
        return nothing, nothing
    end
    if solver == H_X_SOLVER
        return solve_custom_model(model)
    end
end