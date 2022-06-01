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

model, ref_model = get_tm1()

### Determine reference solution
x_ref, u_ref, s_ref, y_ref = DRO.solve_model(ref_model, [2., 2.])
writedlm("output/reference_solution.dat", x_ref, ',')

### Run vanilla solver
DRO.solve_model(
  model, 
  [2., 2.], 
  verbose=DRO.PRINT_AND_WRITE, 
  filename = "output/residuals.dat", 
  z0=zeros(model.nz), 
  v0=zeros(model.nv),
  tol=1e-4
)