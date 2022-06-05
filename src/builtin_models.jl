"""
This file contains some predefined models.
"""

function get_tp1(N :: Int64, alpha :: Float64)
  # Scenario tree
  d = 2; nx = 2; nu = 1
  scen_tree = generate_scenario_tree(N, d, nx, nu)

  # Dynamics: Based on a discretized car model
  T_s = 0.1
  A = [[[1.,0.] [T_s, 1.0 - rand()*T_s]] for _ in 1:d]
  B = [reshape([0., T_s], :, 1) for _ in 1:d]
  dynamics = get_uniform_dynamics(A, B)

  # Cost
  Q = LA.Matrix([2.2 0; 0 3.7])
  R = reshape([3.2], 1, 1)
  cost = get_uniform_cost(Q, R, N)

  # Risk measures
  p_ref = [0.5, 0.5]
  rms = get_uniform_rms_avar(p_ref, alpha, d, N)

  return build_model(scen_tree, cost, dynamics, rms, DYNAMICS_IN_L_SOLVER), build_model(scen_tree, cost, dynamics, rms, MOSEK_SOLVER)
end

function get_tp2(N :: Int64, r :: Float64)
  # Scenario tree
  d = 2; nx = 2; nu = 1
  scen_tree = generate_scenario_tree(N, d, nx, nu)

  # Dynamics: Based on a discretized car model
  T_s = 0.1
  A = [[[1.,0.] [T_s, 1.0 - rand()*T_s]] for _ in 1:d]
  B = [reshape([0., T_s], :, 1) for _ in 1:d]
  dynamics = get_uniform_dynamics(A, B)

  # Cost
  Q = LA.Matrix([2.2 0; 0 3.7])
  R = reshape([3.2], 1, 1)
  cost = get_uniform_cost(Q, R, N)

  # Risk measures
  p_ref = [0.5, 0.5]
  rms = get_uniform_rms_tv(p_ref, r, d, N)

  return build_model(scen_tree, cost, dynamics, rms, DYNAMICS_IN_L_SOLVER), build_model(scen_tree, cost, dynamics, rms, MOSEK_SOLVER)
end

function get_tp1(N :: Int64, alpha :: Float64, p_ref)
  # Scenario tree
  d = 2; nx = 2; nu = 1
  scen_tree = generate_scenario_tree(N, d, nx, nu)

  # Dynamics: Based on a discretized car model
  T_s = 0.1
  A = [[[1.,0.] [T_s, 1.0 - rand()*T_s]] for _ in 1:d]
  B = [reshape([0., T_s], :, 1) for _ in 1:d]
  dynamics = get_uniform_dynamics(A, B)

  # Cost
  Q = LA.Matrix([2.2 0; 0 3.7])
  R = reshape([3.2], 1, 1)
  cost = get_uniform_cost(Q, R, N)

  # Risk measures
  rms = get_uniform_rms_avar(p_ref, alpha, d, N)

  return build_model(scen_tree, cost, dynamics, rms, DYNAMICS_IN_L_SOLVER), build_model(scen_tree, cost, dynamics, rms, MOSEK_SOLVER)
end

function get_tp2(N :: Int64, r :: Float64, p_ref)
  # Scenario tree
  d = 2; nx = 2; nu = 1
  scen_tree = generate_scenario_tree(N, d, nx, nu)

  # Dynamics: Based on a discretized car model
  T_s = 0.1
  A = [[[1.,0.] [T_s, 1.0 - rand()*T_s]] for _ in 1:d]
  B = [reshape([0., T_s], :, 1) for _ in 1:d]
  dynamics = get_uniform_dynamics(A, B)

  # Cost
  Q = LA.Matrix([2.2 0; 0 3.7])
  R = reshape([3.2], 1, 1)
  cost = get_uniform_cost(Q, R, N)

  # Risk measures
  rms = get_uniform_rms_tv(p_ref, r, d, N)

  return build_model(scen_tree, cost, dynamics, rms, DYNAMICS_IN_L_SOLVER), build_model(scen_tree, cost, dynamics, rms, MOSEK_SOLVER)
end