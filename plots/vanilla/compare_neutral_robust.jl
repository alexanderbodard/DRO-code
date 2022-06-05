using DRO, BenchmarkTools, DelimitedFiles, Random

Random.seed!(1234)

Ns = [2, 3, 4, 5, 6, 7]
alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
p_ref = [0.5, 0.5]

global model

######################
### TP1
######################

for (N_i, N) in enumerate(Ns)
  for (alpha_i, alpha) in enumerate(alphas)
    global model, ref_model = get_tp1(N, alpha, p_ref)

    DRO.solve_model(
      model, 
      [1., 1.], 
      verbose=DRO.PRINT_AND_WRITE, 
      path = "logs/",
      filename = "vanilla_neutral_robust_tp1_$(N)_$(alpha_i)", 
      z0=zeros(model.nz), 
      v0=zeros(model.nv),
      tol=1e-6,
      MAX_ITER_COUNT = Int(7.5e5),
      log_stride = 100
    )
  end
end

iterations = zeros(length(Ns), length(alphas))

for (N_i, N) in enumerate(Ns)
  for (alpha_i, alpha) in enumerate(alphas)
    residuals = readdlm("logs/vanilla_neutral_robust_tp1_$(N)_$(alpha_i)_residual.dat", ',')
    iterations[N_i, alpha_i] = length(residuals) * 100
  end
end

open("output/vanilla_neutral_robust_tp1_timings.txt"; write=true) do f
  write(f, "{alpha = 0.1} {alpha = 0.3} {alpha = 0.5} {alpha = 0.7} {alpha = 0.9}\n")
  writedlm(f, iterations[1:end, end:-1:1], ' ') # correct for different alpha definition in text
end

######################
### TP2
######################

for (N_i, N) in enumerate(Ns)
  for (alpha_i, alpha) in enumerate(alphas)
    global model, ref_model = get_tp2(N, alpha, p_ref)

    DRO.solve_model(
      model, 
      [1., 1.], 
      verbose=DRO.PRINT_AND_WRITE, 
      path = "logs/",
      filename = "vanilla_neutral_robust_tp2_$(N)_$(alpha_i)", 
      z0=zeros(model.nz), 
      v0=zeros(model.nv),
      tol=1e-6,
      MAX_ITER_COUNT = Int(7.5e5),
      log_stride = 100
    )
  end
end

for (N_i, N) in enumerate(Ns)
  for (alpha_i, alpha) in enumerate(alphas)
    residuals = readdlm("logs/vanilla_neutral_robust_tp2_$(N)_$(alpha_i)_residual.dat", ',')
    iterations[N_i, alpha_i] = length(residuals) * 100
  end
end

open("output/vanilla_neutral_robust_tp2_timings.txt"; write=true) do f
  write(f, "{r = 0.1} {r = 0.3} {r = 0.5} {r = 0.7} {r = 0.9}\n")
  writedlm(f, iterations, ' ')
end