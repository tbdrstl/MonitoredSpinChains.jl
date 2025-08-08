export compute_anomalous_groundstates


function get_proj_fredkin(L::Int; ancilla::Bool=false) ::Vector{SparseMatrixCSC{ComplexF64, Int64}}
    all_proj = begin
        projectors = [speye(2^(site-1)) ⊗ fredkin ⊗ speye(2^(L-site-2)) for site in 1:L-2]
        push!(projectors, proj(down)⊗speye(2^(L-1)))
        push!(projectors, speye(2^(L-1))⊗proj(up))
        projectors
    end
    if ancilla
        return [proj ⊗ speye(2) for proj in all_proj]
    else
        return all_proj
    end
end

function get_proj_fredkin_pbc(L::Int; ancilla::Bool=false) ::Vector{SparseMatrixCSC{ComplexF64, Int64}}
    all_proj = begin
        projectors = [speye(2^(site-1)) ⊗ fredkin ⊗ speye(2^(L-site-2)) for site in 1:L-2]
        pbc_proj = (speye(2^L) - X ⊗ (speye(2^(L-2)) ⊗ X) - Y ⊗ (speye(2^(L-2)) ⊗ Y) - Z ⊗ (speye(2^(L-2)) ⊗ Z)) * 0.25 
        site_Lminus1 = pbc_proj * (speye(2^(L-2)) ⊗ proj(up) ⊗ speye(2)) + proj(down) ⊗ speye(2^(L-3)) ⊗ Projector |> SparseMatrixCSC{Float64, Int64}
        site_L = Projector ⊗ speye(2^(L-3)) ⊗ proj(up) + pbc_proj * (speye(2) ⊗ proj(down) ⊗ speye(2^(L-2))) |> SparseMatrixCSC{Float64, Int64}
        vcat(projectors, [site_Lminus1, site_L])
    end
    if ancilla
        return [proj ⊗ speye(2) for proj in all_proj]
    else
        return all_proj
    end
end

function get_proj_motzkin(L::Int; ancilla::Bool=false) ::Vector{SparseMatrixCSC{ComplexF64, Int64}}
    all_proj = begin
        projectors = [speye(3^(site-1)) ⊗ motzkin ⊗ speye(3^(L-site-1)) for site in 1:L-1]
        push!(projectors, proj(down1)⊗speye(3^(L-1)))
        push!(projectors, speye(3^(L-1))⊗proj(up1))
        projectors
    end
    if ancilla
        return [proj ⊗ speye(3) for proj in all_proj]
    else
        return all_proj
    end
end

function get_proj_motzkin_pbc(L::Int; ancilla::Bool=false) ::Vector{SparseMatrixCSC{ComplexF64, Int64}}
    all_proj = begin
        projectors = [speye(3^(site-1)) ⊗ motzkin ⊗ speye(3^(L-site-1)) for site in 1:L-1]
        P(v1,v2,v3,v4) = (v2*v4') ⊗ speye(3^(L-2)) ⊗ (v1*v3')
        P(v1,v2) = (v1*v1') ⊗ speye(3^(L-2)) ⊗ (v2*v2')
        pbc_proj = 0.5*(
            P(flat1, up1) - P(flat1, up1, up1, flat1) - P(up1, flat1, flat1, up1) + P(up1, flat1)
        +   P(flat1, down1) + P(down1, flat1) - P(flat1, down1, down1, flat1) - P(down1, flat1, flat1, down1)
        +   P(flat1,flat1) + P(down1,up1) - P(flat1,flat1,up1, down1) - P(up1, down1, flat1, flat1)
        )
        vcat(projectors, [pbc_proj])
    end
    if ancilla
        return [proj ⊗ speye(3) for proj in all_proj]
    else
        return all_proj
    end
end

function get_proj_su2(L::Int64; pbc::Bool=true, ancilla::Bool=false) ::Vector{SparseMatrixCSC{ComplexF64, Int64}}
    all_proj = begin
        projectors = [speye(2^(site-1))⊗ Projector ⊗speye(2^(L-site-1)) for site in 1:L-1]
        if pbc
            pbc_proj = [(speye(2^L) - X ⊗ (speye(2^(L-2)) ⊗ X) - Y ⊗ (speye(2^(L-2)) ⊗ Y) - Z ⊗ (speye(2^(L-2)) ⊗ Z)) * 0.25]
            vcat(projectors, pbc_proj)
        else
            projectors
        end
    end
    if ancilla
        return [proj ⊗ speye(2) for proj in all_proj]
    else
        return all_proj
    end
end

function get_proj_aklt(L::Int64; pbc::Bool=true, ancilla::Bool=false) ::Vector{SparseMatrixCSC{ComplexF64, Int64}}
    all_proj = begin
        projectors = [speye(3^(site-1)) ⊗ Proj1 ⊗ speye(3^(L-site-1)) for site in 1:L-1]
        if pbc 
            u = up1; d = down1; f = flat1
            P(v1,v2,v3,v4) = (v2*v4') ⊗ speye(3^(L-2)) ⊗ (v1*v3')
            P(v1,v2) = (v1*v1') ⊗ speye(3^(L-2)) ⊗ (v2*v2')
            phi2 = P(u,u)
            phi1 = 0.5*(P(u,f) + P(f,u) + P(u,f,f,u) + P(f,u,u,f))
            phi0 = 1/6*(P(u,d) + P(d,u) + 4*P(f,f) + P(u,d,d,u) + P(d,u,u,d) + 2*(P(u,d,f,f) + P(d,u,f,f) + P(f,f,u,d) + P(f,f,d,u)))
            phi_1= 0.5*(P(d,f) + P(f,d) + P(d,f,f,d) + P(f,d,d,f))
            phi_2= P(d,d)
            pbc_proj = speye(3^L) - (phi2 + phi1 + phi0 + phi_1 + phi_2)
            vcat(projectors, [pbc_proj])
        else
            projectors
        end
    end
    if ancilla
        return [proj ⊗ speye(3) for proj in all_proj]
    else
        return all_proj
    end
end

function get_proj0_aklt(L::Int64; pbc::Bool=true, ancilla::Bool=false) ::Vector{SparseMatrixCSC{ComplexF64, Int64}}
    all_proj = begin
        proj0 = 1/sqrt(3)*(up1⊗down1 - flat1 ⊗ flat1 + down1⊗up1)
        proj0 = proj(proj0)
        projectors = [speye(3^(site-1)) ⊗ proj0 ⊗ speye(3^(L-site-1)) for site in 1:L-1]
        if pbc 
            u = up1; d = down1; f = flat1
            P(v1,v2,v3,v4) = (v2*v4') ⊗ speye(3^(L-2)) ⊗ (v1*v3')
            P(v1,v2) = (v1*v1') ⊗ speye(3^(L-2)) ⊗ (v2*v2')
            phi0 = (P(u,d) + P(d,u) + P(f,f) + P(u,d,d,u) + P(d,u,u,d) 
                - P(u,d,f,f) - P(d,u,f,f) - P(f,f,u,d) - P(f,f,d,u))/3.0
            vcat(projectors, [phi0])
        else
            projectors
        end
    end
    if ancilla
        return [proj ⊗ speye(3) for proj in all_proj]
    else
        return all_proj
    end
end

function get_projectors(circuit::Circuit; ancilla::Bool=false) ::Vector{SparseMatrixCSC{ComplexF64, Int64}} 
    trajtype = determine_trajectory(circuit)
    L = circuit.L

    if trajtype == FredkinPBCTrajectory
        projectors = get_proj_fredkin_pbc(L; ancilla=ancilla)
    elseif trajtype == FredkinTrajectory
        projectors = get_proj_fredkin(L; ancilla=ancilla)
    elseif trajtype == MotzkinPBCTrajectory
        projectors = get_proj_motzkin_pbc(L; ancilla=ancilla)
    elseif trajtype == MotzkinTrajectory
        projectors = get_proj_motzkin(L; ancilla=ancilla)
    elseif trajtype == SU2PBCTrajectory
        projectors = get_proj_su2(L; pbc=true, ancilla=ancilla)
    elseif trajtype == SU2Trajectory
        projectors = get_proj_su2(L; pbc=false, ancilla=ancilla)
    elseif trajtype == BiquadraticPBCTrajectory
        projectors = get_proj_aklt(L; ancilla=ancilla)
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
        (_,model) in enumerate(params["model"])

        
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
            model
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
    ancilla = (:AE in circuit.observables)
    
    for trajectoryID in 1:circuit.average
        traj_type = determine_trajectory(circuit)
        traj = traj_type(  trajectoryID = trajectoryID,
                                        circuit = circuit,
                                        current_timestep = current_timestep,
                                        thermalized = thermalized,
                                        observables = observables)
        # check if trajectory has already been computed
        file = trajectory_to_filename(traj)
        computed_file = joinpath(traj.circuit.result_folder, "already_computed", basename(file))
        if !isfile(computed_file)
            push!(trajectories, traj)
        else
            continue
        end
        if state
            trajectories[end].state = circuit.initialState(circuit.L; ancilla=ancilla)
        end

        if projectors
            trajectories[end].projectors = get_projectors(circuit; ancilla=ancilla)
        end

        # if feedbackIdx
        #     trajectories[end].zFeedbackIndices = feedbackIndices(circuit)
        # end

    end
    return trajectories
end

function determine_trajectory(circuit::Circuit)
    traj_type = 0
    ancilla = (:ancilla in circuit.observables) || (:AMI in circuit.observables)

    # check local spin
    if circuit.model == "fredkin"
        traj_type += 1
    elseif circuit.model == "motzkin"
        traj_type += 3
    elseif circuit.model == "su2"
        traj_type += 5
    elseif circuit.model == "biquadratic"
        traj_type += 7
    else
        error("Unknown model type: $(circuit.model)")
    end
    
    # check boundary condition
    if circuit.bc == :obc
        throw(error("Open boundary conditions are not supported for this model."))
        traj_type += 0
    elseif circuit.bc == :pbc
        traj_type += 1
    end

    # assert trajectory type
    if traj_type == 1
        return FredkinTrajectory
    elseif traj_type == 2
        return ancilla ? FredkinPBCTrajectoryA : FredkinPBCTrajectory
    elseif traj_type == 3
        return MotzkinTrajectory
    elseif traj_type == 4
        return ancilla ? MotzkinPBCTrajectoryA : MotzkinPBCTrajectory
    elseif traj_type == 5
        return SU2Trajectory
    elseif traj_type == 6
        return ancilla ? SU2PBCTrajectoryA : SU2PBCTrajectory
    elseif traj_type == 7
        error("Biquadratic only supports PBC")
    elseif traj_type == 8
        return ancilla ? BiquadraticPBCTrajectoryA : BiquadraticPBCTrajectory
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

function compute_missing_parameters!(traj::SpinOneTrajectory)
    if ismissing(traj.projectors)
        ancilla = (:AE in traj.circuit.observables)
        traj.projectors = get_projectors(traj.circuit; ancilla=ancilla)
    end

    if ismissing(traj.state)
        traj.state = traj.circuit.initialState(traj.circuit.L)
    end

    return 
end

function compute_missing_parameters!(traj::SpinHalfTrajectory)
    ancilla = (:AE in traj.circuit.observables) 
    if ismissing(traj.projectors)
        traj.projectors = get_projectors(traj.circuit; ancilla=ancilla)
    end

    if ismissing(traj.state)
        try 
            traj.state = traj.circuit.initialState(traj.circuit.L; ancilla=ancilla)
        catch e
            @warn("No Ancilla version available for this initial state: ", string(traj.circuit.initialState))
            @warn("Trying to continue anyways ...")
            traj.state = traj.circuit.initialState(traj.circuit.L)
        end
    end

    # if ismissing(traj.zFeedbackIndices)
    #     traj.zFeedbackIndices = feedbackIndices(traj.circuit)
    # end

    return 
end

function swapEntries!(x::AbstractVector{T},i::Int,j::Int) where {T <: Union{Float64, ComplexF64}}
    idata = x[i]
    x[i] = x[j]
    x[j] = idata
end

function timesMinusOne!(a::AbstractVector{T}, ind::Vector{I}) where {I <: Integer, T <: Union{Float64, ComplexF64}}
    a[ind] .*= -1.0
    # @fastmath @inbounds @simd for i in ind
    #     a[i] = -a[i]
    # end
end

function feedbackIndices(circuit::Circuit)::Vector{Vector{Int32}}
    @unpack L = circuit 
    feedbackIndices = Vector{Vector{Int32}}(undef, L)
    for site in eachindex(1:L)
        onesDiagonal = diag(speye(2^(site-1)) ⊗ Int32.([1 0; 0 -1]) ⊗ speye(2^(L-site)))
        
        # encouter ancilla observables
        if :AE in circuit.observables
            onesDiagonal = diag(onesDiagonal ⊗ speye(2))
        end
        
        feedbackIndices[site] = findall(!isone, onesDiagonal)
    end
    return feedbackIndices
end

function compute_anomalous_groundstates(;L_min=4, L_max=20)
    if !ispath(anomalousstatepath)
        mkpath(anomalousstatepath)
    end
    for L in L_min:L_max
        if iseven(L)
            if !isfile(joinpath(anomalousstatepath,"anomalousGS$L.bin"))
                write(joinpath(anomalousstatepath,"anomalousGS$L.bin"), anomalous_ground_state(L))
                println("L = $L done")
            end
        end
    end
end