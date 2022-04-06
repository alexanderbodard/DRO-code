using Test
include("../src/scenario_tree.jl")

@testset "Generate scenario trees" begin
    function verify_generated_scenario_tree(N, d, nx, nu, scen_tree)
        gen_scen_tree = generate_scenario_tree(N, d, nx, nu)

        @test scen_tree.anc_mapping == gen_scen_tree.anc_mapping
        @test scen_tree.child_mapping == gen_scen_tree.child_mapping
        @test scen_tree.n_x == gen_scen_tree.n_x
        @test scen_tree.n_u == gen_scen_tree.n_u
        @test scen_tree.n == gen_scen_tree.n
        @test scen_tree.n_non_leaf_nodes == gen_scen_tree.n_non_leaf_nodes
        @test scen_tree.leaf_node_min_index == gen_scen_tree.leaf_node_min_index
        @test scen_tree.leaf_node_max_index == gen_scen_tree.leaf_node_max_index
        @test scen_tree.min_index_per_timestep == gen_scen_tree.min_index_per_timestep
        @test length(scen_tree.node_info) == length(gen_scen_tree.node_info)

        for ind = 1:length(node_info)
            @test scen_tree.node_info[ind].x == gen_scen_tree.node_info[ind].x
            @test scen_tree.node_info[ind].u == gen_scen_tree.node_info[ind].u
            @test scen_tree.node_info[ind].w == gen_scen_tree.node_info[ind].w
            @test scen_tree.node_info[ind].s == gen_scen_tree.node_info[ind].s
        end
    end

    """ 
    N = 3, d = 2, n_x = 2, n_u = 1
    """
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

    N = 3; d = 2; nx = 2; nu = 1
    verify_generated_scenario_tree(N, d, nx, nu, scen_tree)

    """ 
    N = 2, d = 3, n_x = 2, n_u = 1
    """
    node_info = [
        ScenarioTreeNodeInfo(
            collect((i - 1) * 3 + 1 : i * 3),
            i < 2 ? [i] : nothing,
            i > 1 ? (i % 3) + 1 : nothing,
            i,
        ) for i in collect(1:4)
    ]

    scen_tree = ScenarioTree(
        Dict(
            1 => [2, 3, 4],
        ),
        Dict(
            2 => 1,
            3 => 1,
            4 => 1,
        ),
        node_info,
        2,
        1,
        4,
        1,
        2,
        4,
        [1, 2]
    )

    N = 2; d = 3; nx = 2; nu = 1
    verify_generated_scenario_tree(N, d, nx, nu, scen_tree)
end