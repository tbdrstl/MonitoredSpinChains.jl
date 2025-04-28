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

    unitarySteps = ceil(Int, traj.circuit.unitaryRate)
    unitaryTimeEvolProb = traj.circuit.unitaryRate / unitarySteps


    time_step!(traj.circuit, traj, unitaryTimeEvolProb, unitarySteps)
end

function time_step!(circuit::LocalTimestep, traj::Trajectory, unitaryTimeEvolProb::Float64, unitarySteps::Int64) :: Trajectory
    @unpack L, measurement = circuit
    # the actual time evolution

    @show unitaryTimeEvolProb, unitarySteps
    for _ in 1:L
        for __ in 1:unitarySteps
            if rand() < unitaryTimeEvolProb
                random_unitary!(traj.state, circuit, rand(1:L), traj.projectors)
            end
        end

        perturbInThisTimestep = traj.current_timestep <= length(traj.circuit.whenToPerturb) ? traj.circuit.whenToPerturb[traj.current_timestep] : false

        if perturbInThisTimestep
            apply_perturbation!(traj.state, circuit)
        end

        measurement && singlet_meas!(traj, rand(1:L))
    end

    return traj
end

function meas!(traj::Trajectory, projector::SparseArrays.SparseMatrixCSC{ComplexF64, Int64}, site::Int)
    Ppsi = projector * traj.state
    prob = real(dot(traj.state, Ppsi))
    if rand()<prob
        sqrtProb = sqrt(prob)
        traj.state .= Ppsi/sqrtProb

        correct!(state,site)
    else
        traj.state .= (traj.state - Ppsi)/sqrt(1.0-prob)
    end

    return false
end

function correct!(traj::FredkinTrajectory, site::Int)
    L = Int(log2(length(traj.state)))
    # correct the state
    if site < L-1
        traj.state .= (speye(2^(site-1)) ⊗ Z ⊗ speye(2^(L-site))) * traj.state
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
    L = Int(log2(length(traj.state)))
    # correct the state
    site = mod1(site+1, L)
    traj.state .= (speye(2^(site-1)) ⊗ Z ⊗ speye(2^(L-site))) * traj.state

    return 
end

function correct!(traj::MotzkinTrajectory, site::Int)
    L = Int(log(3,length(traj.state)))
    # correct the state
    if site < L-1
        traj.state .= (speye(3^(site-1)) ⊗ sz ⊗ speye(3^(L-site))) * traj.state
    else
        site = mod1(site, L)
        traj.state .= (speye(3^(site-1)) ⊗ Xm ⊗ speye(3^(L-site))) * traj.state
    end
    return 
end

function correct!(traj::MotzkinPBCTrajectory, site::Int)
    L = Int(log(3,length(traj.state)))
    # correct the state
    traj.state .= (speye(3^(site-1)) ⊗ sz ⊗ speye(3^(L-site))) * traj.state
end