using DelimitedFiles, Plots, LaTeXStrings
import LinearAlgebra as LA
pgfplotsx()

Ns = collect(2:15)

"""
Plot the scaling of the parallel implementation
"""

fig = plot(
  xlabel = L"T",
  ylabel = "Execution time (ms)",
  fmt = :pdf,
  legend = false
)

mosek = readdlm("output/mosek_timings.txt", ' ')
plot!(Ns[1:length(mosek)], mosek, yaxis=:log, color=:red, labels = ["Mosek"])
scatter!(Ns[1:length(mosek)], mosek, yaxis=:log, color=:red, labels = [], legend=true)

gurobi = readdlm("output/gurobi_timings.txt", ' ')
plot!(Ns[1:length(gurobi)], gurobi, yaxis=:log, color=:blue, labels = ["Gurobi"])
scatter!(Ns[1:length(gurobi)], gurobi, yaxis=:log, color=:blue, labels = [], legend=true)

ipopt = readdlm("output/ipopt_timings.txt", ' ')
plot!(Ns[1:length(ipopt)], ipopt, yaxis=:log, color=:green, labels = ["IPOPT"])
scatter!(Ns[1:length(ipopt)], ipopt, yaxis=:log, color=:green, labels = [], legend=true)

sequentials = readdlm("../parallel/logs/sequential_timings.txt", ' ')
sequential = sequentials[1:end, 1]
sequential_non_epigraphs = sequentials[1:end, 2]
parallel = readdlm("../parallel/logs/parallel_timings.txt", ' ')

cp = min.(sequential, parallel) * 10

plot!(Ns, cp, yaxis=:log, color=:black, linestyle=:dash, labels = ["CP\n(1000 iterations)"])
scatter!(Ns, cp, yaxis=:log, color=:black, linestyle=:dash, labels = [])

# plot!(Ns, sequential_non_epigraphs, yaxis=:log, color=:green, labels = ["Sequential Routines Only"])
# scatter!(Ns, sequential_non_epigraphs, yaxis=:log, color=:green, labels = [])

# plot!(Ns, parallel, yaxis=:log, color=:red, labels = ["Parallel CP"])
# scatter!(Ns, parallel, yaxis=:log, color=:red, labels = [], legend=true)

filename = "output/generic_solvers.pdf"
savefig(filename)