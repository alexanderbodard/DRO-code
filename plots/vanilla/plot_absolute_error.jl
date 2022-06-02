using DelimitedFiles, Plots, LaTeXStrings
import LinearAlgebra as LA
pgfplotsx()

Ns = [3, 5, 7]
alphas = [0.1, 0.5, 0.9]
STRIDE = 100

fig = plot(
  xlabel = "Iteration",
  ylabel = "Error " * L"\Vert x^k - x^\star \Vert / \Vert x^\star \Vert"
)

colors = [:blue, :green, :red]
linestyles = [:solid, :dash, :dashdot]

for (N_i, N) in enumerate(Ns)
  for (alpha_i, alpha) in enumerate(alphas)
    solution = readdlm("logs/vanilla_tp1_$(N)_$(alpha_i)_x.dat", ',')
    errs = Float64[]
    for i = 1:size(solution)[1]-1
        append!(errs, LA.norm(solution[i, :] .- solution[end, :]) / LA.norm(solution[end, :]))
    end
    plot!(1:STRIDE:(size(solution)[1]-1) * STRIDE, errs, fmt = :png, labels=[L"N = %$(N), \alpha = %$(alpha)"], yaxis=:log, color=colors[N_i], linestyle=linestyles[alpha_i])
  end
end


filename = "output/vanilla_tp1_absolute_error.png"
savefig(filename)