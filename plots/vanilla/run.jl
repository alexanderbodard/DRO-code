"""
This script logs data for the vanilla model solution to the following builtin Test Problems (TPs):
- TP1

This data is written to the output folder and then used to generate the plots in the thesis.
"""

using DRO, BenchmarkTools, DelimitedFiles, Random

Random.seed!(1234)

######################
### TP1
######################

Ns = [3, 5, 7]
alphas = [0.1, 0.5, 0.9]

model, ref_model = get_tp1(3, 0.1)

for N in Ns
  for (alpha_i, alpha) in enumerate(alphas)
    model, _ = get_tp1(N, alpha)

    DRO.solve_model(
      model, 
      [2., 2.], 
      verbose=DRO.PRINT_AND_WRITE, 
      path = "logs/",
      filename = "vanilla_$(N)_$(alpha_i).dat", 
      z0=zeros(model.nz), 
      v0=zeros(model.nv),
      tol=1e-8
    )

    bt = @benchmark DRO.solve_model(
      model, 
      [2., 2.],
      z0=zeros(model.nz), 
      v0=zeros(model.nv),
      tol=1e-8
    )

    t = minimum(bt.times)
    println(t)
  end
end