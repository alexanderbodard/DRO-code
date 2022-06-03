######
# Mosek model
######

using Gurobi, Ipopt

"""
This function returns a Mosek model for the given problem.
"""
function build_mosek_model(scen_tree :: ScenarioTree, cost :: Cost, dynamics :: Dynamics, rms :: Vector{Riskmeasure})
    # Define model, primal variables, epigraph variables and objective
    model = Model(Mosek.Optimizer)
    # model = Model(Gurobi.Optimizer)
    # model = Model(Ipopt.Optimizer)
    set_silent(model)

    @variable(model, x[i=1:scen_tree.n * scen_tree.n_x])
    @variable(model, u[i=1:(length(scen_tree.min_index_per_timestep) - 1) * scen_tree.n_u])
    @variable(model, s[i=1:scen_tree.n * 1])

    @objective(model, Min, s[1])

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
    # TODO: Use tol

    # Initial condition
    @constraint(model, intial_condition[i=1:length(x0)], model[:x][i] .== x0[i])

    optimize!(model)
    return value.(model[:x]), value.(model[:u]), value.(model[:s]), value.(model[:y])
end