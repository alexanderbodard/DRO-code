using DelimitedFiles, Plots, LaTeXStrings, DRO
import LinearAlgebra as LA
pgfplotsx()

"""
Plot the timings as N increases for a fixed time step
"""

Ns = collect(9:-1:2)
alpha = 0.5

fig = plot(
  xlabel = L"N",
  ylabel = "Execution time (ms)",
  fmt = :png,
  legend = false
)

colors = [:blue, :green, :red]
linestyles = [:solid, :dash, :dashdot]

timings = readdlm("logs/vanilla_tp1_timings_fixed_step.txt", ' ')

plot!(Ns, timings, yaxis=:log, color=:blue)
for (N_i, N) in enumerate(Ns)
  scatter!([N], [timings[N_i]], yaxis=:log, color = :blue)
end


filename = "output/vanilla_tp1_timings_fixed_step.png"
savefig(filename)