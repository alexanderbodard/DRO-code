"""
This script logs data for the vanilla model solution to the following builtin Test Problems (TPs):
- TP1

This data is written to the output folder and then used to generate the plots in the thesis.
"""

using DRO, BenchmarkTools, DelimitedFiles, Random
import LinearAlgebra as LA

Random.seed!(123)

######################
### TP1
######################

@eval DRO BISECTION_TOL = 1e-8

N = 3
alpha = 0.1
T = 20
T_s = 0.1

# Scenario tree
d = 2; nx = 2; nu = 1
scen_tree = DRO.generate_scenario_tree(N, d, nx, nu)

# Dynamics: Based on a discretized car model
A = [[[1.,0.] [T_s, 1.0 - rand()*T_s]] for _ in 1:d]
B = [reshape([0., T_s], :, 1) for _ in 1:d]
dynamics = DRO.get_uniform_dynamics(A, B)

# Cost
Q = LA.Matrix([2.2 0; 0 3.7])
R = reshape([3.2], 1, 1)
cost = DRO.get_uniform_cost(Q, R, N)

# Risk measures
p_ref = [0.5, 0.5]
rms = DRO.get_uniform_rms_avar(p_ref, alpha, d, N)

global model = DRO.build_model(scen_tree, cost, dynamics, rms, DRO.DYNAMICS_IN_L_SOLVER, solver_options=DRO.SolverOptions(true))

global x = [1., 1.]

# Log with warm start
for t = 1:T
  if t === 1
    DRO.solve_model(
      model, 
      x, 
      verbose=DRO.PRINT_AND_WRITE, 
      path = "logs/",
      filename = "supermann_tp1_$(t)",
      tol=1e-7,
      MAX_ITER_COUNT = Int(1e4),
      log_stride = 10,
      z0=zeros(model.nz), 
      v0=zeros(model.nv)
    )
  else
    DRO.solve_model(
      model, 
      x, 
      verbose=DRO.PRINT_AND_WRITE, 
      path = "logs/",
      filename = "supermann_tp1_$(t)",
      tol=1e-7,
      MAX_ITER_COUNT = Int(1e4),
      log_stride = 10
    )
  end

  global x = A[1] * x + B[1] * model.z[15]
  global x = x[1:end, 1]
end

global x = [1., 1.]

# Log for comparison
for t = 1:T
  DRO.solve_model(
    model, 
    x, 
    verbose=DRO.PRINT_AND_WRITE, 
    path = "logs/",
    filename = "supermann_tp1_$(t)_ref",
    tol=1e-7,
    MAX_ITER_COUNT = Int(1e4),
    log_stride = 10,
    z0=zeros(model.nz), 
    v0=zeros(model.nv)
  )

  global x = A[1] * x + B[1] * model.z[15]
  global x = x[1:end, 1]
end