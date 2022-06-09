using DelimitedFiles, Plots, LaTeXStrings
import LinearAlgebra as LA
pgfplotsx()

N = 3
alpha = 0.9
T = 20
T_s = 0.1
STRIDE = 10

t_print = 1:T

fig = plot(
  xlabel = "Simulation time step",
  ylabel = "Proportion of the updates that are " * L"K2" * " updates",
  resolution=(800,300)
)

colors = [:blue, :green, :red]
linestyles = [:solid, :dash, :dashdot]
linestyles_text = linestyles[end:-1:1]

k1s = Int[]
k2s = Int[]
for t = 1:T
    if t in t_print
      ks = readdlm("logs/supermann_tp1_$(t)_ks.dat", ',')
      k1 = ks[ks .> 0]
      append!(k1s, length(k1))
      # scatter!(k1[1:end, 1], k1[1:end, 2], fmt = :pdf, color=:red, markersize = 2, legend=false)
      k2 = ks[ks .< 0]
      append!(k2s, length(k2))
      # scatter!(k2[1:end, 1], k2[1:end, 2], fmt = :pdf, color=:blue, markersize = 2, legend=false)
    end
end

scatter!(1:T, k2s ./ (k1s + k2s), color=:red, labels=[], fmt=:pdf)
plot!(1:T, k2s ./ (k1s + k2s), color=:red, labels=["Warm start"], fmt=:pdf)

k1s = Int[]
k2s = Int[]
for t = 1:T
    if t in t_print
      ks = readdlm("logs/supermann_tp1_$(t)_ref_ks.dat", ',')
      k1 = ks[ks .> 0]
      append!(k1s, length(k1))
      # scatter!(k1[1:end, 1], k1[1:end, 2], fmt = :pdf, color=:red, markersize = 2, legend=false)
      k2 = ks[ks .< 0]
      append!(k2s, length(k2))
      # scatter!(k2[1:end, 1], k2[1:end, 2], fmt = :pdf, color=:blue, markersize = 2, legend=false)
    end
end

scatter!(1:T, k2s ./ (k1s + k2s), color=:blue, labels=[], fmt=:pdf)
plot!(1:T, k2s ./ (k1s + k2s), color=:blue, labels=["No warm start"], fmt=:pdf)


filename = "output/supermann_warm_start_tp1_ks_v2.pdf"
savefig(filename)
