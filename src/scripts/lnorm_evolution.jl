using ProximalOperators, Random, JuMP, MosekTools, SparseArrays, Plots, Profile

include("../scenario_tree.jl")
include("../risk_constraints.jl")
include("../dynamics.jl")
include("../cost.jl")

include("../model.jl")
include("../custom_model.jl")
include("../dynamics_in_l_vanilla_model.jl")
include("../dynamics_in_l_supermann_model.jl")
include("../ricatti_vanilla_model.jl")
include("../mosek_model.jl")

import MathOptInterface as MOI
import MathOptSetDistances as MOD
import LinearAlgebra as LA

Random.seed!(1234)

##########################
##########################

function build_model_n(N :: Int64, RICATTI :: Bool, SuperMann :: Bool)
    # Scenario tree
    d = 2; nx = 2; nu = 1
    scen_tree = generate_scenario_tree(N, d, nx, nu)

    # Dynamics: Based on a discretized car model
    T_s = 0.05
    A = [[[1.,0.] [T_s, 1.0 - rand()*T_s]] for _ in 1:d]
    B = [reshape([0., T_s], :, 1) for _ in 1:d]
    dynamics = Dynamics(A, B, nx, nu)

    # Cost: Let's take a quadratic cost, equal at all timesteps
    Q = LA.Matrix([2.2 0; 0 3.7])
    R = reshape([3.2], 1, 1)
    cost = Cost(
        collect([
            Q for _ in 1:N
        ]),
        collect([
            R for _ in 1:N
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

    if RICATTI
        return build_model(scen_tree, cost, dynamics, rms, RICATTI_SOLVER, solver_options=SolverOptions(SuperMann))
    else
        return build_model(scen_tree, cost, dynamics, rms, DYNAMICS_IN_L_SOLVER, solver_options=SolverOptions(SuperMann))
    end
end

######################
Nmax = 9

dyns = zeros(Nmax); ricats = zeros(Nmax)
for n = 2:Nmax
    dyn_in_l_model = build_model_n(n, false, true)
    dyns[n] = dyn_in_l_model.L_norm
end
for n = 2:Nmax
    ricatti_model = build_model_n(n, true, true)
    ricats[n] = ricatti_model.L_norm
end

println(dyns)
println(ricats)

pgfplotsx()
# plot(dyns[2:end], fmt = :png, yaxis=:log, labels=["Dynamics in L"], xlabel="N", ylabel = "|| L ||")
# plot!(ricats[2:end], fmt = :png, yaxis=:log, labels=["Ricatti"])
plot(dyns[2:end] - ricats[2:end], fmt = :png, yaxis=:log, labels=["L difference"], xlabel="N", ylabel = "|| L ||")
filename = "saved_output/L_norm.png"
savefig(filename)

println("-----------")
