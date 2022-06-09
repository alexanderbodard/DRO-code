# DRO-code

The `DRO` module provides a framework for solving Risk-Averse Optimal Control Problems.

### API

To use this framework, first build a model with the `build_model()` function, and then solve it using the `solve_model()` function.
The build step is performed once, offline, whereas the solve step may be performed multiple times, online.

```
build_model(scen_tree :: ScenarioTree, cost :: Cost, dynamics :: Dynamics, rms :: Vector{Riskmeasure}, solver :: Solver; solver_options :: SolverOptions = SolverOptions(false))

solve_model(model :: SOLVER_MODEL, x0 :: Vector{Float64}, tol :: Float64 = 1e-8)
```
Implementations for specific solution methods may accept additional optional arguments.

##### Problem definition

To generate a scenario tree with a constant branching factor $d$ and prediction horizon $N$, you can use the function
```
generate_scenario_tree(N :: Int64, d :: Int64, nx :: Int64, nu :: Int64)
```
More involved scenario trees can be created directly through the constructor.
Similarly, a cost object with all stage cost equal can be obtained through
```
get_uniform_cost(Q :: Matrix{F}, R :: Matrix{F}, N :: I) where {F, I}
```
To impose differing stage costs, use the constructor. Currently only quadratic costs are supported.

Linear dynamics can be imposed by passing the corresponding matrices for each realization to the function
```
get_uniform_dynamics(A :: Vector{Matrix{T}}, B :: Vector{Matrix{T}}) where {T}
```

A `Riskmeasure` object contains a conic representation of the risk measure, such that any conic representable risk measure is compatible with this code.
```
struct Riskmeasure <: AbstractRiskMeasure
    A:: Matrix{Float64}
    B:: Matrix{Float64}
    b:: Vector{Float64}
    C:: ConvexCone
    D:: ConvexCone
end
```

The robust, risk-neutral, AV@R and TV risk measures have been preimplemented.

```
get_uniform_rms_robust(d, N)
get_uniform_rms_risk_neutral(p, d, N)
get_uniform_rms_avar(p, alpha, d, N)
get_uniform_rms_tv(p, r, d, N)
```

### GPU implementation

To parallelize the computations, simply define the `DRO.GPU` variable.
This requires a CUDA GPU.