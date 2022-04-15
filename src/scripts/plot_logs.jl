using DelimitedFiles, Plots
import LinearAlgebra as LA

pgfplotsx()

### Read data
x_supermann = readdlm("output/log_supermann_x.dat", ',')
x_vanilla = readdlm("output/log_vanilla_x.dat", ',')
residual_vanilla = readdlm("output/log_vanilla_residual.dat", ',')
residual_supermann = readdlm("output/log_supermann_residual.dat", ',')
xref = readdlm("output/log_xref.dat", ',')
tau = readdlm("output/log_supermann_tau.dat")

println(issorted(view(residual_vanilla, :), rev=true))


### Plot data

# # x
# plot(x[1:100:end, 1:3], fmt = :png, labels=["x1" "x2" "x3"])
# filename = "output/log_x.png"
# savefig(filename)

# residual norm
plot(residual_vanilla, fmt = :png, labels=["Vanilla"], yaxis=:log)
plot!(residual_supermann, fmt = :png, labels=["Supermann"], yaxis=:log)
filename = "output/log_residual.png"
savefig(filename)

# Absolute error
residues_vanilla = Float64[]
for i = 1:size(x_vanilla)[1]
    append!(residues_vanilla, LA.norm(x_vanilla[i, 1:length(xref)] .- xref) / LA.norm(zeros(length(xref)) .- xref))
end
residues_supermann = Float64[]
for i = 1:size(x_supermann)[1]
    append!(residues_supermann, LA.norm(x_supermann[i, 1:length(xref)] .- xref) / LA.norm(zeros(length(xref)) .- xref))
end
plot(1:length(residual_vanilla), residues_vanilla, fmt = :png, yaxis=:log, labels=["Vanilla"])
plot!(residues_supermann, fmt = :png, yaxis=:log, labels=["Supermann"])
plot!(tau)
filename = "output/log_absolute_error.png"
savefig(filename)

println("")
