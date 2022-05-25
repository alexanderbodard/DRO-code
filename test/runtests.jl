using Test, ProximalOperators, Random, JuMP, MosekTools, SparseArrays, Plots, Profile, DelimitedFiles, ForwardDiff

## Load code
include("../src/scenario_tree.jl")
include("../src/risk_constraints.jl")
include("../src/dynamics.jl")
include("../src/cost.jl")

include("../src/model.jl")
include("../src/custom_model.jl")
include("../src/dynamics_in_l_vanilla_model.jl")
include("../src/dynamics_in_l_supermann_model.jl")
include("../src/ricatti_vanilla_model.jl")
include("../src/mosek_model.jl")

## Load tests

include("scenario_tree.jl")
include("custom_model.jl")
# include("black-box.jl")