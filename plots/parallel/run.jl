"""
This script logs data for the parallelized vanilla model solution to the following builtin Test Problems (TPs):
- TP1

This data is written to the output folder and then used to generate the plots in the thesis.
"""

GPU = true

using DRO, BenchmarkTools, DelimitedFiles, Random

Random.seed!(1234)

######################
### TP1
######################

Ns = [3, 5, 7]
alphas = [0.1, 0.5, 0.9]

global model

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

open("output/parallel_tp1_timings.txt"; write=true) do f
  write(f, "{alpha = 0.1} {alpha = 0.5} {alpha = 0.9}\n")
  writedlm(f, timings * 1e-6, ' ')
end