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
Model in which the problem dynamics Hx = 0 are imposed by including H in the L matrix.
"""
struct DYNAMICS_IN_L_MODEL{T, TT, TTT, U} <: CUSTOM_SOLVER_MODEL
    L :: T
    Ltrans :: TT
    grad_f :: TTT
    prox_hstar_Sigmainv :: U
    L_norm :: Float64
    nz :: Int64
    nv :: Int64
    x_inds :: Vector{Int64}
    u_inds :: Vector{Int64}
    s_inds :: Vector{Int64}
    y_inds :: Vector{Int64}
end

"""
Type union for all available models
"""
const SOLVER_MODEL = Union{MOSEK_MODEL, CUSTOM_SOLVER_MODEL}

"""
Model in which the problem dynamics are imposed through a Ricatti equation
"""
struct RICATTI_MODEL <: CUSTOM_SOLVER_MODEL
    L :: Any
end

"""
Model in which the state variables are eliminated
"""
struct ELIMINATE_STATE_MODEL <: CUSTOM_SOLVER_MODEL
    L :: Any
end

function build_model(scen_tree :: ScenarioTree, cost :: Cost, dynamics :: Dynamics, rms :: Vector{Riskmeasure}, solver :: Solver)
    if solver == MOSEK_SOLVER
        return build_mosek_model(scen_tree, cost, dynamics, rms)
    end
    if solver == DYNAMICS_IN_L_SOLVER
        return build_dynamics_in_l_model(scen_tree, cost, dynamics, rms)     
    end
    error("Building a solver of type $(solver) is not supported.")
end

function solve_model(model :: SOLVER_MODEL, x0 :: Vector{Float64}, tol :: Float64 = 1e-8)
    error("Solving a model of type $(model) is not supported.")
end