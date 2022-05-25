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

pie([15,15,30,60], explode=[0,0.3,0,0], labels=["A","B","C","D"], autopct="%1.1f%%", shadow=true, startangle=90)