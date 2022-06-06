"""
This script logs data for the vanilla model solution to the following builtin Test Problems (TPs):
- TP1

This data is written to the output folder and then used to generate the plots in the thesis.
"""

using DRO, BenchmarkTools, DelimitedFiles, Random

Random.seed!(123)

######################
### TP1
######################

@eval DRO BISECTION_TOL = 1e-8

Ns = [3, 5]
alphas = [0.1, 0.5, 0.9]

global model

# Log Vanilla for comparison
for (N_i, N) in enumerate(Ns)
  for (alpha_i, alpha) in enumerate(alphas)
    global model, ref_model = get_tp1(N, alpha, supermann=false)

    DRO.solve_model(
      model, 
      [1., 1.], 
      verbose=DRO.PRINT_AND_WRITE, 
      path = "logs/",
      filename = "vanilla_tp1_$(N)_$(alpha_i)", 
      z0=zeros(model.nz), 
      v0=zeros(model.nv),
      tol=1e-7,
      MAX_ITER_COUNT = Int(1e6),
      log_stride = 10
    )
  end
end

# Log low memory
for (N_i, N) in enumerate(Ns)
  for (alpha_i, alpha) in enumerate(alphas)
    global model, ref_model = get_tp1(N, alpha, supermann=true)

    DRO.solve_model(
      model, 
      [1., 1.], 
      verbose=DRO.PRINT_AND_WRITE, 
      path = "logs/",
      filename = "supermann_tp1_$(N)_$(alpha_i)", 
      z0=zeros(model.nz), 
      v0=zeros(model.nv),
      tol=1e-7,
      MAX_ITER_COUNT = Int(1e5),
      log_stride = 10
    )
  end
end

Ns = [3]

error("")

# Log sherman morrison
for (N_i, N) in enumerate(Ns)
  for (alpha_i, alpha) in enumerate(alphas)
    global model, ref_model = get_tp1(N, alpha, supermann=true)

    DRO.solve_model(
      model, 
      [1., 1.], 
      verbose=DRO.PRINT_AND_WRITE, 
      path = "logs/",
      filename = "supermann_tp1_sherman_$(N)_$(alpha_i)", 
      z0=zeros(model.nz), 
      v0=zeros(model.nv),
      tol=1e-7,
      MAX_ITER_COUNT = Int(5e4),
      log_stride = 10,
      LOW_MEMORY=false
    )
  end
end

