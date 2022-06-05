using DelimitedFiles, Plots, LaTeXStrings
import LinearAlgebra as LA
pgfplotsx()

######################
### TP1
######################

Ns = [5]
alphas = [0.1, 0.5, 0.9]
alphas_text = alphas[end:-1:1]
STRIDE = 100

fig = plot(
  xlabel = "Iteration",
  ylabel = "Error " * L"\Vert x^k - x^\star \Vert / \Vert x^\star \Vert"
)

colors = [:blue, :green, :red]
linestyles = [:solid, :dash, :dashdot]
linestyles_text = linestyles[end:-1:1]

for (N_i, N) in enumerate(Ns)
  for (alpha_i, alpha) in Iterators.reverse(enumerate(alphas))
    solution = readdlm("logs/supermann_tp1_$(N)_$(alpha_i)_x.dat", ',')
    errs = Float64[]
    for i = 1:size(solution)[1]-1
        append!(errs, LA.norm(solution[i, :] .- solution[end, :]) / LA.norm(solution[end, :]))
    end
    if alpha == 0.5
      plot!(1:STRIDE:(size(solution)[1]-1) * STRIDE, errs, fmt = :png, labels=[L"N = %$(N), \alpha = %$(alphas_text[alpha_i])"], yaxis=:log, color=colors[N_i], linestyle=linestyles_text[alpha_i])
    end
  end
end

for (N_i, N) in enumerate(Ns)
  for (alpha_i, alpha) in Iterators.reverse(enumerate(alphas))
    solution = readdlm("../vanilla/logs/vanilla_tp1_$(N)_$(alpha_i)_x.dat", ',')
    errs = Float64[]
    for i = 1:size(solution)[1]-1
        append!(errs, LA.norm(solution[i, :] .- solution[end, :]) / LA.norm(solution[end, :]))
    end
    if alpha == 0.5
      plot!(1:STRIDE:(size(solution)[1]-1) * STRIDE, errs, fmt = :png, labels=[L"N = %$(N), \alpha = %$(alphas_text[alpha_i])"], yaxis=:log, color=:red)
    end
  end
end


filename = "output/supermann_tp1_absolute_error.png"
savefig(filename)

# ######################
# ### Sherman Morrison
# ######################

Ns = [3, 5]
alphas = [0.1, 0.5, 0.9]
alphas_text = alphas[end:-1:1]

fig = plot(
  xlabel = "Iteration",
  ylabel = "Error " * L"\Vert x^k - x^\star \Vert / \Vert x^\star \Vert"
)

colors = [:blue, :green, :red]
linestyles = [:solid, :dash, :dashdot]
linestyles_text = linestyles[end:-1:1]

for (N_i, N) in enumerate(Ns)
  for (alpha_i, alpha) in Iterators.reverse(enumerate(alphas))
    solution = readdlm("logs/supermann_tp1_sherman_$(N)_$(alpha_i)_x.dat", ',')
    errs = Float64[]
    for i = 1:size(solution)[1]-1
        append!(errs, LA.norm(solution[i, :] .- solution[end, :]) / LA.norm(solution[end, :]))
    end
    plot!(1:STRIDE:(size(solution)[1]-1) * STRIDE, errs, fmt = :png, labels=[L"N = %$(N), \alpha = %$(alphas_text[alpha_i])"], yaxis=:log, color=colors[N_i], linestyle=linestyles_text[alpha_i])
  end
end


filename = "output/supermann_tp1_sherman_absolute_error.png"
savefig(filename)