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
  fmt = :png,
  legend = false
)

sequentials = readdlm("logs/sequential_timings.txt", ' ')
sequential = sequentials[1:end, 1]
sequential_non_epigraphs = sequentials[1:end, 2]

plot!(Ns, sequential, yaxis=:log, color=:blue, labels = ["Sequential CP"])
scatter!(Ns, sequential, yaxis=:log, color=:blue, labels = [])

plot!(Ns, sequential_non_epigraphs, yaxis=:log, color=:green, labels = ["Sequential Routines Only"])
scatter!(Ns, sequential_non_epigraphs, yaxis=:log, color=:green, labels = [])

parallel = readdlm("logs/parallel_timings.txt", ' ')

plot!(Ns, parallel, yaxis=:log, color=:red, labels = ["Parallel CP"])
scatter!(Ns, parallel, yaxis=:log, color=:red, labels = [], legend=true)

filename = "output/parallel_vs_sequential.png"
savefig(filename)

fig = plot(
  xlabel = L"T",
  ylabel = "Execution time (ms)",
  fmt = :png,
  legend = false
)

sequentials = readdlm("logs/sequential_timings.txt", ' ')
sequential = sequentials[1:end, 1]
sequential_non_epigraphs = sequentials[1:end, 2]

plot!(Ns, sequential - sequential_non_epigraphs, yaxis=:log, color=:blue, labels = ["Sequential epigraph projections"])
scatter!(Ns, sequential - sequential_non_epigraphs, yaxis=:log, color=:blue, labels = [])

parallel = readdlm("logs/parallel_timings.txt", ' ')

plot!(Ns, parallel - sequential_non_epigraphs, yaxis=:log, color=:red, labels = ["Parallel epigraph projections"])
scatter!(Ns, parallel - sequential_non_epigraphs, yaxis=:log, color=:red, labels = [], legend=true)

filename = "output/parallel_speedup.png"
savefig(filename)