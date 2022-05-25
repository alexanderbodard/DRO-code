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

"""
Returns rms for a constant branching factor d and all risk mappings defined as risk robust.

A = I_d
B = [1_d'; - 1_d']
b = [1; -1]
"""
function get_uniform_rms_robust(d, N)
  return get_uniform_rms(
    LA.I(d),
    vcat([1. for _ in 1:d]', [-1. for _ in 1:d]'),
    [1.; -1.],
    ConvexCone([MOI.Nonnegatives(d)]),
    ConvexCone([MOI.Nonnegatives(2)]),
    d,
    N
  )
end

"""
Returns rms for a constant branching factor d and all risk mappings defined as avar.

A = I_d
B = [I_d; 1_d'; - 1_d']
b = [p / alpha; 1; -1]
"""
function get_uniform_rms_avar(p, alpha, d, N)
  return get_uniform_rms(
    LA.I(d),
    vcat(LA.I(d), [1. for _ in 1:d]', [-1. for _ in 1:d]'),
    [p / alpha; 1.; -1.],
    ConvexCone([MOI.Nonnegatives(d)]),
    ConvexCone([MOI.Nonnegatives(d+2)]),
    d,
    N
  )
end

function get_uniform_rms_risk_neutral(p, d, N)
  return get_uniform_rms_avar(p, 1., d, N)
end