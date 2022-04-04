using Test

include("../src/scenario_tree.jl")
include("../src/dynamics.jl")
include("../src/custom_model.jl")

@testset "Dynamics constraint 4e" begin
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

    L_II, L_JJ, L_VV = construct_L_4e(scen_tree, dynamics, length(x) + length(u))
    println(minimum(L_JJ), ", ", maximum(L_JJ))
    println(L_JJ)
    H = sparse(L_II, L_JJ, L_VV, (scen_tree.n_x * scen_tree.n - 1), scen_tree.n * scen_tree.n_x + scen_tree.n_non_leaf_nodes * scen_tree.n_u)


    @test 5 == 5
    @test 1 + 2 == 3
end