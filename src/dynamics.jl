function run_trajectory!(traj::Trajectory)
    traj.observables = get_observables(traj.circuit)

    load_existing_trajectory_data!(traj)
    compute_missing_parameters!(traj)
    time_evolve!(traj)
    remove_excess_data!(traj)
    # GC.gc(true)
end

function time_evolve!(traj::Trajectory)
    thermalize!(traj)
    for timestep in traj.current_timestep:traj.circuit.meas_steps*traj.circuit.meas_every
        time_step!(traj)
        get_observables!(traj)
        traj.current_timestep += 1
        save_trajectory(traj)
        # GC.gc()
    end
    GC.gc(true)
end

function thermalize!(traj::Trajectory)
    if traj.thermalized
        return
    end
    for timestep in traj.current_timestep:traj.circuit.thermalizationSteps
        time_step!(traj)
        traj.current_timestep += 1
    end    
    traj.current_timestep = 1

    traj.thermalized = true
    return traj
end

function time_step!(traj::Trajectory) :: Trajectory
    # unitarySteps = ceil(Int, traj.circuit.unitaryRate)
    # unitaryTimeEvolProb = traj.circuit.unitaryRate / unitarySteps


    time_step!(traj.circuit, traj)#, unitaryTimeEvolProb, unitarySteps)
end

function time_step!(circuit::Circuit, traj::Trajectory)#, unitaryTimeEvolProb::Float64, unitarySteps::Int64) :: Trajectory
    @unpack L, measurement = circuit
    measurement && meas!(traj, rand(1:L))
    return traj
end

function meas!(traj::Trajectory, site::Int)
    Ppsi = traj.projectors[site] * traj.state
    prob = real(dot(traj.state, Ppsi))
    if rand()<prob
        sqrtProb = sqrt(prob)
        traj.state .= Ppsi/sqrtProb

        correct!(traj,site)
    else
        traj.state .= (traj.state - Ppsi)/sqrt(1.0-prob)
    end

    return 
end

function correct!(traj::SU2Trajectory, site::Int)
    controlPsiZ!(traj.state, traj.zFeedbackIndices[site])
end

function correct!(traj::SU2PBCTrajectory, site::Int)
    controlPsiZ!(traj.state, traj.zFeedbackIndices[site])
end

function correct!(traj::FredkinTrajectory, site::Int)
    L = Int(log2(length(traj.state)))
    # correct the state
    if site < L-1
        controlPsiZ!(traj.state, traj.zFeedbackIndices[site])
    elseif site == L-1
        traj.state .= (X ⊗ speye(2^(L-1))) * traj.state
    elseif site == L
        traj.state .= (speye(2^(L-1)) ⊗ X) * traj.state
    else
        error("site out of bounds")
    end
    return 
end

function correct!(traj::FredkinPBCTrajectory, site::Int)
    # correct the state
    site = mod1(site+1, traj.circuit.L)
    controlPsiZ!(traj.state, traj.zFeedbackIndices[site])

    return 
end

function correct!(traj::AKLTPBCTrajectory, site::Int)
    L = traj.circuit.L
    # correct the state
    # if rand(Bool) # correct with Z
        traj.state .= (speye(3^(site-1)) ⊗ exp_z1 ⊗ speye(3^(L-site))) * traj.state
    # else # correct with XX
    #     if site < L-1
    #         traj.state .= (speye(3^(site-1)) ⊗ Xm ⊗ Xm ⊗ speye(3^(L-site-1))) * traj.state
    #     else
    #         site = mod1(site, L)
    #         traj.state .= (Xm ⊗ speye(3^(L-2)) ⊗ Xm) * traj.state
    #     end

    # end
    # if site < L-1
    #     traj.state .= (speye(3^(site-1)) ⊗ sz ⊗ speye(3^(L-site))) * traj.state
    # else
    #     site = mod1(site, L)
    #     traj.state .= (speye(3^(site-1)) ⊗ exp_x1 ⊗ speye(3^(L-site))) * traj.state
    # end
    return 
end

function correct!(traj::MotzkinTrajectory, site::Int)
    L = traj.circuit.L
    # correct the state
    if 1 < site < L-1
        traj.state .= (speye(3^(site-1)) ⊗ exp_z1 ⊗ speye(3^(L-site))) * traj.state
    else
        site = mod1(site, L)
        traj.state .= (speye(3^(site-1)) ⊗ Px ⊗ speye(3^(L-site))) * traj.state
    end
    return 
end

function correct!(traj::MotzkinPBCTrajectory, site::Int)
    L = traj.circuit.L
    # correct the state
    traj.state .= (speye(3^(site-1)) ⊗ exp_z1 ⊗ speye(3^(L-site))) * traj.state
end

function controlPsiZ!(state::AbstractVector{T}, feedbackIndices::Vector{I}) where {I<:Integer, T<:Union{Float64, ComplexF64}}
    timesMinusOne!(state, feedbackIndices)
end