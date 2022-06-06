function cost_projection_cuda!(Q_bars, vv_workspace, vvv_workspace, inds_4d, nQ)# Q, vector that contains (among other) x | t, workspace
  """
  for scen_ind = 1:length(inds_4d) ÷ nQ # for each scenario
      epigraph_bisection!(
          Q_bars,
          vv_workspace, 
          vv_workspace[inds_4d[scen_ind * nQ] + 1],
          vvv_workspace,
          inds_4d,
          scen_ind,
          nQ
      )
  end
  """
  scen_ind = convert(Int64, CUDA.threadIdx().x)
  block_ind = convert(Int64, CUDA.blockIdx().x)
  scen_ind = scen_ind + 32 * (block_ind - 1)
  epigraph_bisection!(
    Q_bars,
      vv_workspace, 
      vv_workspace[inds_4d[scen_ind * nQ] + 1],
      vvv_workspace,
      inds_4d,
      scen_ind,
      nQ
  )
  
  return nothing
end

function cuda_projection!(
  model :: DYNAMICS_IN_L_MODEL,
)
  # 4a
  for ind in model.inds_4a
    model.vv_workspace[ind] = MOD.projection_on_set(MOD.DefaultDistance(), view(model.vv_workspace, ind), MOI.Nonpositives(2)) # TODO: Fix polar cone
  end

  # 4b
  model.vv_workspace[model.inds_4b] = MOD.projection_on_set(MOD.DefaultDistance(), view(model.vv_workspace, model.inds_4b), MOI.Nonpositives(4)) # TODO: Fix polar cone
  
  # 4c
  for (i, ind) in enumerate(model.inds_4c)
    b_bar = model.b_bars[i]
    vv = view(model.vv_workspace, ind)
    dot_p = LA.dot(vv, b_bar)
    if dot_p > 0
        model.vv_workspace[ind] = vv - dot_p / LA.dot(b_bar, b_bar) * b_bar
    end
  end
  
  # 4d: Compute cost projection
  #cost_projection_cuda!(model.Q_bars, model.vv_workspace, model.vvv_workspace, model.inds_4d, model.nQ)
  copyto!(model.vv_workspace_cuda, model.vv_workspace)

  #CUDA.@sync begin
    @cuda threads=(32) blocks=16 cost_projection_cuda!(model.Q_bars, model.vv_workspace_cuda, model.vvv_workspace, model.inds_4d, model.nQ)
  #end

  copyto!(model.vv_workspace, model.vv_workspace_cuda)

  # 4e: Dynamics
  @simd for ind in model.inds_4e
      @inbounds @fastmath model.vv_workspace[ind] = 0.
  end

  # Initial condition
  model.vv_workspace[end - length(model.x0) + 1 : end] = model.x0
end

function prox_g!(
  model :: DYNAMICS_IN_L_VANILLA_MODEL,
  arg :: Vector{Float64},
  sigma :: Float64
)
  # TODO: Write a bit more efficient
  # TODO: Remove model.x0 as argument

  # vv_workspace = arg / sigma
  copyto!(model.vv_workspace, arg)
  LA.BLAS.scal!(model.nv, 1. / sigma, model.vv_workspace, stride(model.vv_workspace, 1))

  # Result is stored in vv_workspace
  CUDA.@sync cuda_projection!(model)
  
  # LA.BLAS.axpy!(-sigma, model.vv_workspace, arg)
  # copyto!(model.vbar, arg)
  @simd for i = 1:model.nv
      @inbounds @fastmath model.vbar[i] = arg[i] - sigma * model.vv_workspace[i]
  end
end
