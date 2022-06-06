using DelimitedFiles, Plots, LaTeXStrings
import LinearAlgebra as LA
pgfplotsx()

######################
### TP1
######################

N = 3
alpha = 0.9
T = 20
T_s = 0.1
STRIDE = 10

t_print = 1:T

fig = plot(
  xlabel = "Iteration",
  ylabel = L"\pm (\tau + t)",
  resolution=(800,300)
)

colors = [:blue, :green, :red]
linestyles = [:solid, :dash, :dashdot]
linestyles_text = linestyles[end:-1:1]

for t = 1:T
    if t in t_print
      ks = readdlm("logs/supermann_tp1_$(t)_ks.dat", ',')
      ks = hcat(1:STRIDE:length(ks)*STRIDE, ks)
      k1 = ks[ks[:, 2] .> 0, :]
      k1[1:end, 2] .+= t - 1
      scatter!(k1[1:end, 1], k1[1:end, 2], fmt = :png, color=:red, markersize = 2, legend=false)
      k2 = ks[ks[:, 2] .< 0, :]
      k2[1:end, 2] .-= t-1
      scatter!(k2[1:end, 1], k2[1:end, 2], fmt = :png, color=:blue, markersize = 2, legend=false)
    end
end


filename = "output/supermann_warm_start_tp1_ks.png"
savefig(filename)

fig = plot(
  xlabel = "Iteration",
  ylabel = L"\pm (\tau + t)",
  resolution=(800,300)
)

colors = [:blue, :green, :red]
linestyles = [:solid, :dash, :dashdot]
linestyles_text = linestyles[end:-1:1]

for t = 1:T
    if t in t_print
      ks = readdlm("logs/supermann_tp1_$(t)_ref_ks.dat", ',')
      ks = hcat(1:STRIDE:length(ks)*STRIDE, ks)
      k1 = ks[ks[:, 2] .> 0, :]
      k1[1:end, 2] .+= t - 1
      scatter!(k1[1:end, 1], k1[1:end, 2], fmt = :png, color=:red, markersize = 2, legend=false)
      k2 = ks[ks[:, 2] .< 0, :]
      k2[1:end, 2] .-= t-1
      scatter!(k2[1:end, 1], k2[1:end, 2], fmt = :png, color=:blue, markersize = 2, legend=false)
    end
end


filename = "output/supermann_warm_start_tp1_ks_ref.png"
savefig(filename)