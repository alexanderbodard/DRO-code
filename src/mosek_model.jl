######
# Mosek model
######

"""
This function returns a Mosek model for the given problem.
"""
function build_mosek_model(scen_tree :: ScenarioTree, cost :: Cost, dynamics :: Dynamics, rms :: Vector{Riskmeasure})
    # Define model, primal variables, epigraph variables and objective
    model = Model(Mosek.Optimizer)
    set_silent(model)

    @variable(model, x[i=1:scen_tree.n * scen_tree.n_x])
    @variable(model, u[i=1:scen_tree.n_non_leaf_nodes * scen_tree.n_u])
    @variable(model, s[i=1:scen_tree.n * 1])

    @objective(model, Min, s[1])

    # TODO: Initial condition should be moved to the call step
    @constraint(model, intial_condition[i=1:1], x[i] .== [2.][i])

    # Impose cost
    impose_cost(model, scen_tree, cost)

    # Impose dynamics
    impose_dynamics(model, scen_tree, dynamics)

    # Impose risk measure epigraph constraints
    add_risk_epi_constraints(model, scen_tree, rms)

    return model
end

"""
This function solves a Mosek model
"""
function solve_model(model :: Model, x0 :: Vector{Float64}, tol :: Float64 = 1e-8)
    # TODO: Use x0
    # TODO: Use tol

    optimize!(model)
    return value.(model[:x]), value.(model[:u]), value.(model[:s]), value.(model[:y])
end