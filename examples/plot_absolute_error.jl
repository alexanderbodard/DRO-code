using DelimitedFiles, Plots
import LinearAlgebra as LA
pgfplotsx()

reference_solution = readdlm("output/reference_solution.dat", ',')
solution = readdlm("output/x.dat", ',')

errs = Float64[]
for i = 1:size(solution)[1]-1
    append!(errs, LA.norm(solution[i, :] .- solution[end, :]) / LA.norm(solution[end, :]))
end

plot(errs, fmt = :png, labels=["Vanilla"], yaxis=:log)

filename = "output/absolute_error.png"
savefig(filename)