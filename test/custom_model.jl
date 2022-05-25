using Test
import LinearAlgebra as LA

@testset "p_norm" begin
    for _ = 1:100
        nz = 30; nv = 100
        L = rand(nv, nz); sigma = rand(); gamma = rand();
        P = [[1/gamma * LA.I(nz) -L']; [-L 1/sigma * LA.I(nv)]]

        rz = rand(nz); rv = rand(nv);

        p_norm_verif = LA.dot(vcat(rz, rv), P * vcat(rz, rv))
        @test isapprox(p_norm_verif, p_norm(rz, rv, rz, rv, L, gamma, sigma))
        @test isapprox(p_norm_verif, p_dot(vcat(rz, rv), vcat(rz, rv), P))
    end
end

@testset "prox_f and psi" begin
    Q = LA.Matrix([2.2 0; 0 3.7])
    Qdiag = [2.2, 3.7]

    for _ = 1:1e3
        x = rand(2); gamma = rand(); s = rand()

        f = ProximalOperators.Quadratic(Q, zeros(2))
        p, t = ProximalOperators.prox(f, x, gamma)
        @test isapprox(prox_f_copy(Qdiag, gamma, x), p)
        @test isapprox(psi_copy(Qdiag, gamma, x, s), 0.5 * p' * Q * p - gamma - s)
    end
end

# @testset "epigraph projection" begin
#     Q = LA.Matrix([2.2 0; 0 3.7])
#     Qdiag = [2.2, 3.7]

#     for _ = 1:1e1
#         x = rand(2); t = rand(); xx = copy(x); tt = copy(t)
#         p, s = epigraph_qcqp(Q, x, t)
#         # (p, s) \in epigraph
#         @test 0.5 * p' * Q * p <= s

#         if 0.5 * x' * Q * x <= t
#             @test isapprox(x, p)
#             @test isapprox(t, s)
#         end

#         pp, ss = epigraph_bisection(Qdiag, x, t)
#         # (pp, ss) \in epigraph
#         @test 0.5 * pp' * Q * pp <= ss

#         if 0.5 * x' * Q * x <= t
#             @test isapprox(x, pp)
#             @test isapprox(t, ss)
#         end
#         if 0.5 * x' * Q * x > t
#             # When (x, t) is not on the epigraph, the projection must be on the boundary
#             @test isapprox(0.5 * p' * Q * p, s)
#             @test isapprox(0.5 * pp' * Q * pp, ss)
#         end

#         gamma_star = ss - t
#         if gamma_star > 0
#             p_test = prox_f_copy(Qdiag, s - t, x)
#             isapprox(p_test, p)
#         end

#         # println("x: $(x)")
#         # @test isapprox(pp, p, rtol=1e-6)
#         # println("t: $(t)")
#         # @test isapprox(ss, s, rtol=1e-6)

#         # println("Difference: ", (ss - s) / s)

#         # Variables x and t have not been altered
#         @test isapprox(xx, x, atol=1e-16)
#         @test isapprox(tt, t, atol=1e-16)
#     end
# end