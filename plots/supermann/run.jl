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
      tol=1e-6,
      MAX_ITER_COUNT = Int(2.5e4),
      log_stride = 10
    )
  end
end

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
      tol=1e-6,
      MAX_ITER_COUNT = Int(5e4),
      log_stride = 10,
      LOW_MEMORY=false
    )
  end
end
# """
# # timings
# timings = zeros(length(Ns), length(alphas))

# for (N_i, N) in enumerate(Ns)
#   for (alpha_i, alpha) in enumerate(alphas)
#     global model, ref_model = get_tp1(N, alpha)

#     bt = @benchmark DRO.solve_model(
#       model, 
#       [1., 1.],
#       z0=zeros(model.nz), 
#       v0=zeros(model.nv),
#       tol=1e-6,
#       MAX_ITER_COUNT = Int(7.5e5)
#     )

#     t = minimum(bt.times)
#     timings[N_i, alpha_i] = t
#   end
# end

# open("output/vanilla_tp1_timings.txt"; write=true) do f
#   write(f, "{alpha = 0.1} {alpha = 0.5} {alpha = 0.9}\n")
#   writedlm(f, timings[1:end, end:-1:1] * 1e-6, ' ')
# end
# """
# ######################
# ### TP2
# ######################
# """
# Ns = [3, 5]
# alphas = [0.1, 0.5, 0.9]

# global model

# # Log
# for (N_i, N) in enumerate(Ns)
#   for (alpha_i, alpha) in enumerate(alphas)
#     global model, ref_model = get_tp2(N, alpha)

#     DRO.solve_model(
#       model, 
#       [1., 1.], 
#       verbose=DRO.PRINT_AND_WRITE, 
#       path = "logs/",
#       filename = "vanilla_tp2_$(N)_$(alpha_i)", 
#       z0=zeros(model.nz), 
#       v0=zeros(model.nv),
#       tol=1e-6,
#       MAX_ITER_COUNT = Int(7.5e5),
#       log_stride = 100
#     )
#   end
# end

# # timings
# timings = zeros(length(Ns), length(alphas))

# for (N_i, N) in enumerate(Ns)
#   for (alpha_i, alpha) in enumerate(alphas)
#     global model, ref_model = get_tp2(N, alpha)

#     bt = @benchmark DRO.solve_model(
#       model, 
#       [1., 1.],
#       z0=zeros(model.nz), 
#       v0=zeros(model.nv),
#       tol=1e-6,
#       MAX_ITER_COUNT = Int(7.5e5)
#     )

#     t = minimum(bt.times)
#     timings[N_i, alpha_i] = t
#   end
# end

# open("output/vanilla_tp2_timings.txt"; write=true) do f
#   write(f, "{alpha = 0.1} {alpha = 0.5} {alpha = 0.9}\n")
#   writedlm(f, timings[1:end, end:-1:1] * 1e-6, ' ')
# end
# """