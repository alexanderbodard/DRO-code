using DelimitedFiles, Plots, LaTeXStrings
pgfplotsx()

######################
### TP1
######################

Ns = [3, 5, 7]
alphas = [0.1, 0.5, 0.9]
alphas_text = alphas[end:-1:1]
STRIDE = 100

fig = plot(
  xlabel = "Iteration",
  ylabel = "Residual " * L"\Vert R x \Vert"
)

colors = [:blue, :green, :red]
linestyles = [:solid, :dash, :dashdot]

linestyles_text = linestyles[end:-1:1]

for (N_i, N) in enumerate(Ns)
  for (alpha_i, alpha) in Iterators.reverse(enumerate(alphas))
    residuals = readdlm("logs/vanilla_tp1_$(N)_$(alpha_i)_residual.dat", ',')
    plot!(1:STRIDE:length(residuals)*STRIDE, residuals, fmt = :pdf, labels=[L"T = %$(N), \alpha = %$(alphas_text[alpha_i])"], yaxis=:log, color=colors[N_i], linestyle=linestyles_text[alpha_i])
  end
end

filename = "output/vanilla_tp1_residual.pdf"
savefig(filename)

######################
### TP2
######################

Ns = [3, 5, 7]
alphas = [0.1, 0.5, 0.9]
STRIDE = 100

fig = plot(
  xlabel = "Iteration",
  ylabel = "Residual " * L"\Vert R x \Vert"
)

colors = [:blue, :green, :red]
linestyles = [:solid, :dash, :dashdot]

for (N_i, N) in enumerate(Ns)
  for (alpha_i, alpha) in enumerate(alphas)
    residuals = readdlm("logs/vanilla_tp2_$(N)_$(alpha_i)_residual.dat", ',')
    plot!(1:STRIDE:length(residuals)*STRIDE, residuals, fmt = :pdf, labels=[L"T = %$(N), r = %$(alpha)"], yaxis=:log, color=colors[N_i], linestyle=linestyles[alpha_i])
  end
end

filename = "output/vanilla_tp2_residual.pdf"
savefig(filename)