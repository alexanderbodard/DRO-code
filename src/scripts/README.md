This folder contains some scripts, meant to be run separately.

`warm_start.jl` is meant to verify the influence of a warm start (like in an MPC context) on the number of iterations of the primal dual algorithm.

`multilevel_start.jl` is meant to verify if a multilevel startup procedure could be beneficial. Specifically, the idea is that if for some large horizon $N$ the primal dual algorithm requires many iterations, then it might well be beneficial to first compute the solution for smaller values of $N$ and warm start the larger problem based on the solution to these smaller problems.