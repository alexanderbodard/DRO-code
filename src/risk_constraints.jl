###
# Risk constraints
###

struct ConvexCone
    subcones:: Array{MOI.AbstractVectorSet}
end

abstract type AbstractRiskMeasure end 

struct Riskmeasure <: AbstractRiskMeasure
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

#####################################################
# Exposed API funcions
#####################################################

"""

"""
function get_uniform_rms(A, B, b, C, D, d, N)
  return [
        Riskmeasure(
            A,
            B,
            b,
            C,
            D
        ) for _ in 1:(d^(N - 1) - 1) / (d - 1)
    ]
end

function get_uniform_rms_risk_neutral()
  # Todo
end