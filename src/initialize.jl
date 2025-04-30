
function get_proj_fredkin(L::Int)
    projectors = [speye(2^(site-1)) ⊗ fredkin ⊗ speye(2^(L-site-2)) for site in 1:L-2]
    push!(projectors, proj(down)⊗speye(2^(L-1)))
    push!(projectors, speye(2^(L-1))⊗proj(up))
    return projectors
end

function get_proj_fredkin_pbc(L::Int)
    projectors = [speye(2^(site-1)) ⊗ fredkin ⊗ speye(2^(L-site-2)) for site in 1:L-2]
    
    pbc_proj = (speye(2^L) - X ⊗ (speye(2^(L-2)) ⊗ X) - Y ⊗ (speye(2^(L-2)) ⊗ Y) - Z ⊗ (speye(2^(L-2)) ⊗ Z)) * 0.25
    
    site_Lminus1 = pbc_proj * (speye(2^(L-2)) ⊗ proj(up) ⊗ speye(2)) + proj(down) ⊗ speye(2^(L-3)) ⊗ Projector
    site_L = Projector ⊗ speye(2^(L-3)) ⊗ proj(up) + pbc_proj * (speye(2) ⊗ proj(down) ⊗ speye(2^(L-2)))
    return vcat(projectors, [site_Lminus1, site_L])
end

function get_proj_motzkin(L::Int)
    projectors = [speye(3^(site-1)) ⊗ motzkin ⊗ speye(3^(L-site-1)) for site in 1:L-1]
    push!(projectors, proj(downm)⊗speye(3^(L-1)))
    push!(projectors, speye(3^(L-1))⊗proj(upm))
    return projectors
end

function get_proj_motzkin_pbc(L::Int)
    projectors = [speye(3^(site-1)) ⊗ motzkin ⊗ speye(3^(L-site-1)) for site in 1:L-1]
    
    pbc_proj = 0.5 * (proj(upm⊗spzeros(3^(L-2))⊗flatm - flatm⊗spzeros(3^(L-2))⊗upm)
    + proj(downm⊗spzeros(3^(L-2))⊗flatm - (flatm |> Vector{ComplexF64})⊗spzeros(3^(L-2))⊗downm)
    + proj(upm⊗spzeros(3^(L-2))⊗downm - flatm⊗spzeros(3^(L-2))⊗flatm))
    
    return vcat(projectors, [pbc_proj])
end

function get_projectors(circuit::Circuit)
    trajtype = determine_trajectory(circuit)
    L = circuit.L

    if trajtype == FredkinPBCTrajectory
        projectors = get_proj_fredkin_pbc(L)
    elseif trajtype == FredkinTrajectory
        projectors = get_proj_fredkin(L)
    elseif trajtype == MotzkinPBCTrajectory
        projectors = get_proj_motzkin_pbc(L)
    elseif trajtype == MotzkinTrajectory
        projectors = get_proj_motzkin(L)
    else
        error("Unknown trajectory type.")
    end
    return projectors
end

function create_simulation(params::Dict; testmode::Bool=false)
    # test if all parameters are given

    for param in required_params
        if !haskey(params, param)
            throw(ArgumentError("Parameter $param is missing."))
        end
    end

    # test if all observables are legal_observables
    for obs in params["observables"]
        if !(obs in legal_observables)
            throw(ArgumentError("Observable $obs is not a legal observable."))
        end
    end

    # for unitarySetup in params["unitarySetup"]
    #     if !(unitarySetup in legal_unitarySetups)
    #         throw(ArgumentError("Unitary setup $unitarySetup is not a legal unitary setup."))
    #     end
    # end

    for feedback in params["feedback"]
        if !(feedback in legal_feedbacks)
            throw(ArgumentError("Feedback $feedback is not a legal feedback."))
        end
    end

    # create Simulation struct and fill it with Circuits
    vector_of_circuits = Vector{Circuit}(undef, 0)

    for (_,systemSize) in enumerate(params["systemSize"]),
        (_,meas_steps) in enumerate(params["meas_steps"]),
        (_,average) in enumerate(params["average"]),
        # (_,unitaryRate) in enumerate(params["unitaryRate"]),
        # (_,unitarySetup) in enumerate(params["unitarySetup"]),
        (_,bc) in enumerate(params["bc"]),
        (_,initialState) in enumerate(params["initialState"]),
        (_,measurement) in enumerate(params["measurement"]),
        (_,feedback) in enumerate(params["feedback"]),
        (_,trajectories_averaged) in enumerate(params["trajectories_averaged"]),
        (_,thermalizationSteps) in enumerate(params["thermalizationSteps"]),
        (_,meas_every) in enumerate(params["meas_every"]),
        (_,local_spin) in enumerate(params["local_spin"])

        
        push!(vector_of_circuits, Circuit(
            systemSize,
            meas_steps(systemSize),
            average,
            # unitaryRate,
            # unitarySetup,
            bc,
            initialState,
            measurement,
            feedback,
            params["result_folder"],
            params["observables"],
            trajectories_averaged,
            thermalizationSteps(systemSize),
            meas_every(systemSize),
            local_spin
        ))
    end

    unique!(vector_of_circuits)
    check_if_circuit_is_already_computed!(vector_of_circuits; testmode=testmode)

    return Simulation(params["name"], vector_of_circuits, params)
end

function get_trajectories_from_simulation(sim::Simulation; state::Bool=false, projectors::Bool=false, feedbackIdx::Bool=false)
    trajectories = Vector{Trajectory}(undef, 0)
    for circuit in sim.params
        traj_circuit = get_trajectories_from_circuit(circuit; state=state, projectors=projectors, feedbackIdx=feedbackIdx)
        push!(trajectories,traj_circuit...)
    end
    return trajectories
end

function get_trajectories_from_circuit(circuit::Circuit; state::Bool=false, projectors::Bool=false, feedbackIdx::Bool=false)
    trajectories = Vector{Trajectory}(undef, 0)
    observables = get_observables(circuit)
    current_timestep = 1
    thermalized = ifelse(circuit.thermalizationSteps == 0, true, false)
    
    for trajectoryID in 1:circuit.average
        traj_type = determine_trajectory(circuit)
        push!(trajectories, traj_type(  trajectoryID = trajectoryID,
                                        circuit = circuit,
                                        current_timestep = current_timestep,
                                        thermalized = thermalized,
                                        observables = observables))
        if state
            trajectories[end].state = circuit.initialState(circuit.L)
        end

        if projectors
            trajectories[end].projectors = get_projectors(circuit)
        end

    end
    return trajectories
end

function determine_trajectory(circuit::Circuit)
    traj_type = 0

    # check local spin
    if circuit.local_spin == 0.5
        traj_type += 0
    elseif isone(circuit.local_spin)
        traj_type += 2
    end
    
    # check boundary condition
    if circuit.bc == :obc
        traj_type += 0
    elseif circuit.bc == :pbc
        traj_type += 1
    end

    # assert trajectory type
    if traj_type == 0
        return FredkinTrajectory
    elseif traj_type == 1
        return FredkinPBCTrajectory
    elseif traj_type == 2
        return MotzkinTrajectory
    elseif traj_type == 3
        return MotzkinPBCTrajectory
    else
        error("Assertion of trajectory type failed.")
    end
end

function check_if_circuit_is_already_computed!(vector_of_circuits::Vector{Circuit}; testmode::Bool=false)
    if testmode
        return
    end

    indexlist = zeros(Int, length(vector_of_circuits))
    for (i, circuit) in enumerate(vector_of_circuits)
        if isfile(circuit_to_filename(circuit, 0; average=true))
            indexlist[i] = i
        end
    end
    filter!(!iszero, indexlist)
    splice!(vector_of_circuits, indexlist)
    return 
end

function compute_missing_parameters!(traj::Trajectory)
    if ismissing(traj.projectors)
        traj.projectors = get_projectors(traj.circuit)
    end

    if ismissing(traj.state)
        traj.state = traj.circuit.initialState(traj.circuit.L)
    end

    return 
end