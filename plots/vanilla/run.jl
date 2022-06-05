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

Ns = [3, 5]
alphas = [0.1, 0.5, 0.9]

global model

# Log
for (N_i, N) in enumerate(Ns)
  for (alpha_i, alpha) in enumerate(alphas)
    global model, ref_model = get_tp1(N, alpha)

    DRO.solve_model(
      model, 
      [1., 1.], 
      verbose=DRO.PRINT_AND_WRITE, 
      path = "logs/",
      filename = "vanilla_tp1_$(N)_$(alpha_i)", 
      z0=zeros(model.nz), 
      v0=zeros(model.nv),
      tol=1e-6,
      MAX_ITER_COUNT = Int(7.5e5),
      log_stride = 100
    )
  end
end
"""
# timings
timings = zeros(length(Ns), length(alphas))

for (N_i, N) in enumerate(Ns)
  for (alpha_i, alpha) in enumerate(alphas)
    global model, ref_model = get_tp1(N, alpha)

    bt = @benchmark DRO.solve_model(
      model, 
      [1., 1.],
      z0=zeros(model.nz), 
      v0=zeros(model.nv),
      tol=1e-6,
      MAX_ITER_COUNT = Int(7.5e5)
    )

    t = minimum(bt.times)
    timings[N_i, alpha_i] = t
  end
end

open("output/vanilla_tp1_timings.txt"; write=true) do f
  write(f, "{alpha = 0.1} {alpha = 0.5} {alpha = 0.9}\n")
  writedlm(f, timings[1:end, end:-1:1] * 1e-6, ' ')
end
"""
# # Spy plot
# model, ref_model = get_tp1(3, 0.5)
# Plots.spy(model.L, legend=nothing)
# savefig("output/tp1_spy.pdf")

# # L norms
# model, ref_model = get_tp1(3, 0.5)
# sigmas = [0.99 * 2.0^(-i) for i = 0:10] / sqrt(model.L_norm)
# norms = zeros(length(sigmas))
# for (sigma_i, sigma) in enumerate(sigmas)
#   norms[sigma_i] = model.L_norm * ((2.0^(sigma_i-1))^2)

#   DRO.solve_model(
#     model, 
#     [1., 1.], 
#     verbose=DRO.PRINT_AND_WRITE, 
#     path = "logs/",
#     filename = "vanilla_tp1_norms_$(sigma_i)", 
#     z0=zeros(model.nz), 
#     v0=zeros(model.nv),
#     tol=1e-6,
#     MAX_ITER_COUNT = Int(7.5e5),
#     log_stride = 10,
#     sigma = sigma,
#     gamma = sigma
#   )
# end
# open("logs/vanilla_tp1_lnorms.txt"; write=true) do f
#   writedlm(f, norms, ' ')
# end

# # Timings for fixed stepsize
# Ns = collect(9:-1:2)
# timings = zeros(length(Ns))
# alpha = 0.5
# gamma = 0.0; sigma = 0.0

# for (N_i, N) in enumerate(Ns)
#   global model, ref_model = get_tp1(N, alpha)
#   println("Benchmarking N = $(N)")

#   if N_i == 1
#     gamma = 0.99 / sqrt(model.L_norm)
#     sigma = gamma
#   end

#   bt = @benchmark DRO.solve_model(
#     model, 
#     [1., 1.],
#     z0=zeros(model.nz), 
#     v0=zeros(model.nv),
#     tol=1e-6,
#     MAX_ITER_COUNT = 10,
#     gamma = gamma,
#     sigma = sigma
#   )

#   t = minimum(bt.times)
#   timings[N_i] = t
# end
# open("logs/vanilla_tp1_timings_fixed_step.txt"; write=true) do f
#   writedlm(f, timings * 1e-6, ' ')
# end

######################
### TP2
######################

Ns = [3, 5]
alphas = [0.1, 0.5, 0.9]

global model

# Log
for (N_i, N) in enumerate(Ns)
  for (alpha_i, alpha) in enumerate(alphas)
    global model, ref_model = get_tp2(N, alpha)

    DRO.solve_model(
      model, 
      [1., 1.], 
      verbose=DRO.PRINT_AND_WRITE, 
      path = "logs/",
      filename = "vanilla_tp2_$(N)_$(alpha_i)", 
      z0=zeros(model.nz), 
      v0=zeros(model.nv),
      tol=1e-6,
      MAX_ITER_COUNT = Int(7.5e5),
      log_stride = 100
    )
  end
end
"""
# timings
timings = zeros(length(Ns), length(alphas))

for (N_i, N) in enumerate(Ns)
  for (alpha_i, alpha) in enumerate(alphas)
    global model, ref_model = get_tp2(N, alpha)

    bt = @benchmark DRO.solve_model(
      model, 
      [1., 1.],
      z0=zeros(model.nz), 
      v0=zeros(model.nv),
      tol=1e-6,
      MAX_ITER_COUNT = Int(7.5e5)
    )

    t = minimum(bt.times)
    timings[N_i, alpha_i] = t
  end
end

open("output/vanilla_tp2_timings.txt"; write=true) do f
  write(f, "{alpha = 0.1} {alpha = 0.5} {alpha = 0.9}\n")
  writedlm(f, timings[1:end, end:-1:1] * 1e-6, ' ')
end
"""