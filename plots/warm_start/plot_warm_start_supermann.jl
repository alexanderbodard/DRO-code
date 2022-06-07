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

MAX_ITER = Int(1e3) # DIVIDE BY STRIDE

fig = plot(
  xlabel = "Iteration",
  ylabel = "Residual " * L"\Vert R x \Vert"
)

colors = [:blue, :green, :red]
linestyles = [:solid, :dash, :dashdot]

linestyles_text = linestyles[end:-1:1]

lengths = Int[]

res = Float64[]
for t = 1:T
  residuals = readdlm("logs/supermann_tp1_$(t)_ref_residual.dat", ',')
  append!(res, residuals[2:end])
  append!(lengths, length(residuals))
end
plot!(1:STRIDE:length(res)*STRIDE, res, fmt = :pdf, labels=["No warm start"], yaxis=:log, color=:blue)

res = Float64[]
for t = 1:T
  residuals = readdlm("logs/supermann_tp1_$(t)_residual.dat", ',')
  append!(res, residuals[2:end], residuals[end] * ones(lengths[t] - length(residuals)))
end
plot!(1:STRIDE:length(res)*STRIDE, res, fmt = :pdf, labels=["Warm start"], yaxis=:log, color=:red)

filename = "output/supermann_tp1_warm_start.pdf"
savefig(filename)