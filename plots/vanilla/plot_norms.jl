using DelimitedFiles, Plots, LaTeXStrings, DRO
import LinearAlgebra as LA
pgfplotsx()

"""
Plot the L norm impact on the number of iterations for fixed N
"""

STRIDE = 10
N = 3
alpha = 0.5

fig = plot(
  xlabel = L"\Vert L \Vert",
  ylabel = "Number of iterations",
  fmt = :png,
  legend = false
)

colors = [:blue, :green, :red]
linestyles = [:solid, :dash, :dashdot]

norms = readdlm("logs/vanilla_tp1_lnorms.txt", ' ')

ys = zeros(length(norms))
for i = 1:length(norms)
  solution = readdlm("logs/vanilla_tp1_norms_$(i)_x.dat", ',')
  iterations = STRIDE * length(solution)
  scatter!([norms[i]], [iterations], xaxis=:log, yaxis=:log, color = :blue)
  ys[i] = iterations
end
plot!(norms, ys, xaxis=:log, yaxis=:log, color=:blue)


filename = "output/vanilla_tp1_lnorms_a.png"
savefig(filename)

"""
Plot the L norm for increasing N
"""

Nmax = 9
norms = zeros(length(2:Nmax))
for N = 2 : Nmax
  model, ref_model = get_tp1(N, 0.5)
  norms[N-1] = model.L_norm
end

fig = plot(
  xlabel = L"N",
  ylabel = L"\Vert L \Vert",
  fmt = :png,
  legend = false
)

plot!(2:Nmax, norms, yaxis=:log, color=:blue, legend=false)

filename = "output/vanilla_tp1_lnorms_b.png"
savefig(filename)