###
# Model
###

"""
Enum to be used to select one of the solver implementations
"""
@enum Solver begin
	MOSEK_SOLVER = 1
	DYNAMICS_IN_L_SOLVER = 2
	RICATTI_SOLVER = 3
	ELIMINATE_STATE_SOLVER = 4
end

"""
Abstract type for all the custom models
"""
abstract type CUSTOM_SOLVER_MODEL end

const MOSEK_MODEL = Model

"""
Type union for all available models
"""
const SOLVER_MODEL = Union{MOSEK_MODEL, CUSTOM_SOLVER_MODEL}

"""
Model in which the problem dynamics Hx = 0 are imposed by including H in the L matrix.
"""
struct DYNAMICS_IN_L_VANILLA_MODEL{T} <: CUSTOM_SOLVER_MODEL
    L :: T
    L_norm :: Float64
    nz :: Int64
    nv :: Int64
    x_inds :: Vector{Int64}
    u_inds :: Vector{Int64}
    s_inds :: Vector{Int64}
    y_inds :: Vector{Int64}
    inds_4a :: Vector{Union{UnitRange{Int64}, Int64}}
    inds_4b :: UnitRange{Int64}
    inds_4c :: Vector{Union{UnitRange{Int64}, Int64}}
    b_bars :: Vector{Vector{Float64}}
    inds_4d :: Vector{Union{UnitRange{Int64}, Int64}}
    Q_bars :: Vector{Vector{Float64}}
    inds_4e :: UnitRange{Int64}
    workspace_vec :: Vector{Float64}
    z_workspace :: Vector{Float64}
    v_workspace :: Vector{Float64}
    vv_workspace :: Vector{Float64}
    vvv_workspace :: Vector{Float64}
    z :: Vector{Float64}
    v :: Vector{Float64}
    rz :: Vector{Float64}
    rv :: Vector{Float64}
    zbar :: Vector{Float64}
    vbar :: Vector{Float64}
    x0 :: Vector{Float64}
end

"""
TODO
"""
struct DYNAMICS_IN_L_SUPERMANN_MODEL{T} <: CUSTOM_SOLVER_MODEL
    L :: T
    L_norm :: Float64
    nz :: Int64
    nv :: Int64
    x_inds :: Vector{Int64}
    u_inds :: Vector{Int64}
    s_inds :: Vector{Int64}
    y_inds :: Vector{Int64}
    inds_4a :: Vector{Union{UnitRange{Int64}, Int64}}
    inds_4b :: UnitRange{Int64}
    inds_4c :: Vector{Union{UnitRange{Int64}, Int64}}
    b_bars :: Vector{Vector{Float64}}
    inds_4d :: Vector{Union{UnitRange{Int64}, Int64}}
    Q_bars :: Vector{Any}
    inds_4e :: UnitRange{Int64}
    workspace_vec :: Vector{Float64}
    x_workspace :: Vector{Float64}
    v_workspace :: Vector{Float64}
end

# Union type for all models with dynamics in L
const DYNAMICS_IN_L_MODEL = Union{DYNAMICS_IN_L_VANILLA_MODEL, DYNAMICS_IN_L_SUPERMANN_MODEL}

"""
Model in which the problem dynamics are imposed through a Ricatti equation
"""
struct RICATTI_VANILLA_MODEL{T} <: CUSTOM_SOLVER_MODEL
    L :: T
    L_norm :: Float64
    nz :: Int64
    nv :: Int64
    x_inds :: Vector{Int64}
    u_inds :: Vector{Int64}
    s_inds :: Vector{Int64}
    y_inds :: Vector{Int64}
    inds_4a :: Vector{Union{UnitRange{Int64}, Int64}}
    inds_4b :: UnitRange{Int64}
    inds_4c :: Vector{Union{UnitRange{Int64}, Int64}}
    b_bars :: Vector{Vector{Float64}}
    inds_4d :: Vector{Union{UnitRange{Int64}, Int64}}
    Q_bars :: Vector{Any}
    inds_4e :: UnitRange{Int64}
    workspace_vec :: Vector{Float64}
    x_workspace :: Vector{Float64}
    v_workspace :: Vector{Float64}
end

"""
Model in which the problem dynamics are imposed through a Ricatti equation
"""
struct RICATTI_SUPERMANN_MODEL <: CUSTOM_SOLVER_MODEL
    L :: Any
end

const RICATTI_MODEL = Union{RICATTI_VANILLA_MODEL, RICATTI_SUPERMANN_MODEL}

"""
Model in which the state variables are eliminated
"""
struct ELIMINATE_STATE_MODEL <: CUSTOM_SOLVER_MODEL
    L :: Any
end

struct SolverOptions
    SuperMann :: Bool
end

function build_model(scen_tree :: ScenarioTree, cost :: Cost, dynamics :: Dynamics, rms :: Vector{Riskmeasure}, solver :: Solver; solver_options :: SolverOptions = SolverOptions(false))
    if solver == MOSEK_SOLVER
        return build_mosek_model(scen_tree, cost, dynamics, rms)
    end
    if solver == DYNAMICS_IN_L_SOLVER
        if solver_options.SuperMann
            return build_dynamics_in_l_supermann_model(scen_tree, cost, dynamics, rms)
        else
            return build_dynamics_in_l_vanilla_model(scen_tree, cost, dynamics, rms)
        end     
    end
    if solver == RICATTI_SOLVER
        return build_ricatti_vanilla_model(scen_tree, cost, dynamics, rms)
    end
    error("Building a solver of type $(solver) is not supported.")
end

function solve_model(model :: SOLVER_MODEL, x0 :: Vector{Float64}, tol :: Float64 = 1e-8)
    error("Solving a model of type $(model) is not supported.")
end