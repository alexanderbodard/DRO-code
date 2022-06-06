using DelimitedFiles, Plots, LaTeXStrings
import LinearAlgebra as LA
pgfplotsx()

######################
### TP1
######################

Ns = [3]
alphas = [0.1, 0.5, 0.9]
alphas_text = alphas[end:-1:1]
STRIDE = 10

ALPHA = 0.9

fig = plot(
  xlabel = "Iteration",
  ylabel = L"\tau",
  # xlims=(0, 2.5e5)
  resolution=(800,300)
)

colors = [:blue, :green, :red]
linestyles = [:solid, :dash, :dashdot]
linestyles_text = linestyles[end:-1:1]

for (N_i, N) in enumerate(Ns)
  for (alpha_i, alpha) in Iterators.reverse(enumerate(alphas))
    ks = readdlm("logs/supermann_tp1_sherman_$(N)_$(alpha_i)_ks.dat", ',')
    ks = hcat(1:STRIDE:length(ks)*STRIDE, ks)
    println(size(ks))
    k1 = ks[ks[:, 2] .> 0, :]
    if alpha == ALPHA
      scatter!(k1[1:end, 1], k1[1:end, 2], fmt = :png, labels=["Educated update"], color=:red, markersize = 2)
    end
    k2 = ks[ks[:, 2] .< 0, :]
    if alpha == ALPHA
      scatter!(k2[1:end, 1], k2[1:end, 2], fmt = :png, labels=["GKM update"], color=:blue, markersize = 2)
    end
  end
end


filename = "output/supermann_tp1_ks_sherman_$(ALPHA).png"
savefig(filename)

fig = plot(
  xlabel = "Iteration",
  ylabel = L"\tau",
  # xlims=(0, 2.5e5)
)

colors = [:blue, :green, :red]
linestyles = [:solid, :dash, :dashdot]
linestyles_text = linestyles[end:-1:1]

for (N_i, N) in enumerate(Ns)
  for (alpha_i, alpha) in Iterators.reverse(enumerate(alphas))
    ks = readdlm("logs/supermann_tp1_$(N)_$(alpha_i)_ks.dat", ',')
    ks = hcat(1:STRIDE:length(ks)*STRIDE, ks)
    k1 = ks[ks[:, 2] .> 0, :]
    if alpha == ALPHA
      scatter!(k1[1:end, 1], k1[1:end, 2], fmt = :png, labels=["Educated update"], color=:red, markersize = 2)
    end
    k2 = ks[ks[:, 2] .< 0, :]
    if alpha == ALPHA
      scatter!(k2[1:end, 1], k2[1:end, 2], fmt = :png, labels=["GKM update"], color=:blue, markersize = 2)
    end
  end
end


filename = "output/supermann_tp1_ks_$(ALPHA).png"
savefig(filename)