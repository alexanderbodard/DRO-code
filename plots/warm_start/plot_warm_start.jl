using DelimitedFiles, Plots, LaTeXStrings
pgfplotsx()

######################
### TP1
######################

N = 3
alpha = 0.9
T = 20
T_s = 0.1
STRIDE = 10

fig = plot(
  xlabel = "Iteration",
  ylabel = "Residual " * L"\Vert R x \Vert"
)

colors = [:blue, :green, :red]
linestyles = [:solid, :dash, :dashdot]

linestyles_text = linestyles[end:-1:1]

res = Float64[]
for t = 1:T
  residuals = readdlm("logs/vanilla_tp1_$(t)_residual.dat", ',')
  append!(res, residuals)
end
plot!(1:STRIDE:length(res)*STRIDE, res, fmt = :pdf, labels=["Warm start"], yaxis=:log, color=:red)

res = Float64[]
for t = 1:T
  residuals = readdlm("logs/vanilla_tp1_$(t)_ref_residual.dat", ',')
  append!(res, residuals)
end
plot!(1:STRIDE:length(res)*STRIDE, res, fmt = :pdf, labels=["No warm start"], yaxis=:log, color=:blue)

filename = "output/vanilla_tp1_warm_start.pdf"
savefig(filename)