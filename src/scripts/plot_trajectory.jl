###
# Problem definition
###
using ProximalOperators, Random, JuMP, MosekTools, SparseArrays, Plots, Profile, DelimitedFiles, ForwardDiff

include("../scenario_tree.jl")
include("../risk_constraints.jl")
include("../dynamics.jl")
include("../cost.jl")

include("../model.jl")
include("../custom_model.jl")
include("../dynamics_in_l_vanilla_model.jl")
include("../dynamics_in_l_supermann_model.jl")
include("../mosek_model.jl")

import MathOptInterface as MOI
import MathOptSetDistances as MOD
import LinearAlgebra as LA

##########################
# Mosek reference implementation
##########################

###
# Problem statement
###

# Scenario tree
N = 5; d = 2; nx = 1; nu = 1
scen_tree = generate_scenario_tree(N, d, nx, nu)

# Dynamics: Based on a discretized car model
Ts = 5
A = [rand() + .2 for _ in 1:d]
B = [rand() * Ts for _ in 1:d]

x0 = 2.

xs = zeros(2^N, N+1)
us = rand(N)

for i = 1:2^N
    xs[i, 1] = x0
end

# for i = 1:2^N
#     node = 1
#     for j = 2:N
#         node += i * 2^(j-2)
#         w = scen_tree.node_info[node].w
#         println(i, ", ", j)
#         xs[i, j] = A[w] * xs[i, j-1] + B[w] * us[j-1]
#     end
# end

for i = 1:2^N
    if i <= 2^(N-1) # i in 1:16
        xs[i, 2] = A[1] * xs[i, 1] + B[1] * us[1]
    else
        xs[i, 2] = A[2] * xs[i, 1] + B[2] * us[1]
    end
end

for i = 1:2^N
    if i <= 2^(N-2) || i in 2^(N-1)+1:2^(N-1)+2^(N-2) # i in 1:8 || i in 17:24
        xs[i, 3] = A[1] * xs[i, 2] + B[1] * us[2]
    else
        xs[i, 3] = A[2] * xs[i, 2] + B[2] * us[2]
    end
end

for i = 1:2^N
    if i in 1:4 || i in 9:12 || i in 17:20 || i in 25:28
        xs[i, 4] = A[1] * xs[i, 3] + B[1] * us[3]
    else
        xs[i, 4] = A[2] * xs[i, 3] + B[2] * us[3]
    end
end

for i = 1:2^N
    if i%4 in 1:2
        xs[i, 5] = A[1] * xs[i, 4] + B[1] * us[4]
    else
        xs[i, 5] = A[2] * xs[i, 4] + B[2] * us[4]
    end
end

for i = 1:2^N
    if i%2 === 1
        xs[i, 6] = A[1] * xs[i, 5] + B[1] * us[5]
    else
        xs[i, 6] = A[2] * xs[i, 5] + B[2] * us[5]
    end
end

x = [xs[1, 1]]
append!(x, [xs[1, 2], xs[end, 2]]) # 2
append!(x, [xs[1, 3], xs[9, 3], xs[17, 3], xs[25, 3]]) # 4
append!(x, [xs[1, 4], xs[5, 4], xs[9, 4], xs[13, 4], xs[17, 4], xs[21, 4], xs[25, 4], xs[29, 4]]) # 8
append!(x, xs[1, 5], xs[3, 5], xs[5, 5], xs[7, 5], xs[9, 5], xs[11, 5], xs[13, 5], xs[15, 5], xs[17, 5], xs[19, 5], xs[21, 5], xs[23, 5], xs[25, 5], xs[27, 5], xs[29, 5], xs[31, 5]) # 16
append!(x, xs[:, 6])
ts = [0. 1. 1. 2. 2. 2. 2. 3. 3. 3. 3. 3. 3. 3. 3. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 5. 5. 5. 5. 5. 5. 5. 5. 5. 5. 5. 5. 5. 5. 5. 5. 5. 5. 5. 5. 5. 5. 5. 5. 5. 5. 5. 5. 5. 5. 5. 5.]'

pgfplotsx()
scatter(ts, x, fmt=:png, marker = (4, 0.2, :orange), label="", ylabel="x", xlabel="t")
filename = string("output/trajectory.png")
savefig(filename)



writedlm("output/trajectory.dat", xs, ',') 