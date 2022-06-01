using DelimitedFiles, Plots
pgfplotsx()

residuals = readdlm("output/residuals.dat", ',')
plot(residuals, fmt = :png, labels=["Vanilla"], yaxis=:log)

filename = "output/residual.png"
savefig(filename)