function cost_projection!(Q_bars, vv_workspace, vvv_workspace, inds_4d, nQ)# Q, vector that contains (among other) x | t, workspace
  for scen_ind = 1:length(inds_4d) ÷ nQ
      # ind = inds_4d[(scen_ind - 1) * nQ + 1 : scen_ind * nQ]
      ind = inds_4d[(scen_ind - 1) * nQ + 1] : inds_4d[scen_ind * nQ]
      vv_workspace[ind[end] + 1] = epigraph_bisection!(
          view(Q_bars, (scen_ind - 1) * nQ + 1 : scen_ind * nQ ),
          view(vv_workspace, ind), 
          vv_workspace[ind[end] + 1],
          view(vvv_workspace, ind)
      )
  end

  nothing
end

function projection!(
  model :: DYNAMICS_IN_L_MODEL,
)
  # 4a
  # for ind in model.inds_4a
    # model.vv_workspace[ind] = MOD.projection_on_set(MOD.DefaultDistance(), view(model.vv_workspace, ind), MOI.Nonpositives(2)) # TODO: Fix polar cone
  # end
  model.vv_workspace[model.inds_4a] = MOD.projection_on_set(MOD.DefaultDistance(), view(model.vv_workspace, model.inds_4a), MOI.Nonpositives(2))

  # 4b
  model.vv_workspace[model.inds_4b] = MOD.projection_on_set(MOD.DefaultDistance(), view(model.vv_workspace, model.inds_4b), MOI.Nonpositives(2)) # TODO: Fix polar cone
  
  # 4c
  for (i, ind) in enumerate(model.inds_4c)
    b_bar = model.b_bars[i]
    vv = view(model.vv_workspace, ind)
    dot_p = LA.dot(vv, b_bar)

    # copyto!(model.vvv_workspace, 1, model.vv_workspace, ind[1], ind[end])
    # dot_p = 0
    # for j = 1:length(ind)
    #   @inbounds @fastmath dot_p += model.vvv_workspace[j] + b_bar[j]
    # end

    if dot_p > 0
        dot_p /= LA.dot(b_bar, b_bar)
        b_bar .*= dot_p
        # @simd for j = 1:length(ind)
        #   @inbounds @fastmath model.vv_workspace[ind[j]] = model.vvv_workspace[j] - b_bar[j]
        # end
        @simd for j = 1:length(ind)
          @inbounds @fastmath model.vv_workspace[ind[j]] = vv[j] - b_bar[j]
        end
        # model.vv_workspace[ind] = vv - dot_p / LA.dot(b_bar, b_bar) * b_bar
    end
  end
  
  # 4d: Compute cost projection
  cost_projection!(model.Q_bars, model.vv_workspace, model.vvv_workspace, model.inds_4d, model.nQ)
  
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
  projection!(model)
  
  # LA.BLAS.axpy!(-sigma, model.vv_workspace, arg)
  # copyto!(model.vbar, arg)
  @simd for i = 1:model.nv
      @inbounds @fastmath model.vbar[i] = arg[i] - sigma * model.vv_workspace[i]
  end
end