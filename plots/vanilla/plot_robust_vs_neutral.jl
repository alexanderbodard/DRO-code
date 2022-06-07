using DelimitedFiles, Plots, LaTeXStrings, DRO
import LinearAlgebra as LA
pgfplotsx()

N = 3;
p_ref = [0.3, 0.7]

alpha = 0.01
model, _ = get_tp1(N, alpha, p_ref)
DRO.solve_model(model, [2., 2.], z0=zeros(model.nz), v0=zeros(model.nv), MAX_ITER_COUNT=1000000, tol=1e-6)
p = histogram(abs.(model.v[model.inds_4d]), bins=10.0 .^ (-20:2), xaxis=:log, xlims=(1e-20, 100), ylims=(0, 20), legend=false, xlabel=L"v_i", ylabel="Bin count")
filename = "output/vanilla_robust_histogram.pdf"
savefig(filename)

alpha = 0.99
model, _ = get_tp1(N, alpha, p_ref)
DRO.solve_model(model, [2., 2.], z0=zeros(model.nz), v0=zeros(model.nv), MAX_ITER_COUNT=1000000, tol=1e-6)
p = histogram(abs.(model.v[model.inds_4d]), bins=10.0 .^ (-20:2), xaxis=:log, xlims=(1e-20, 100), ylims=(0, 40), legend=false, xlabel=L"v_i", ylabel="Bin count")
display(p)
filename = "output/vanilla_neutral_histogram.pdf"
savefig(filename);