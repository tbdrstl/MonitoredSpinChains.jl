export compute_anomalous_groundstates


function get_proj_fredkin(L::Int) ::Vector{SparseMatrixCSC{ComplexF64, Int64}} 
    projectors = [speye(2^(site-1)) ⊗ fredkin ⊗ speye(2^(L-site-2)) for site in 1:L-2]
    push!(projectors, proj(down)⊗speye(2^(L-1)))
    push!(projectors, speye(2^(L-1))⊗proj(up))
    return projectors
end

function get_proj_fredkin_pbc(L::Int) ::Vector{SparseMatrixCSC{ComplexF64, Int64}} 
    projectors = [speye(2^(site-1)) ⊗ fredkin ⊗ speye(2^(L-site-2)) for site in 1:L-2]
    
    pbc_proj = (speye(2^L) - X ⊗ (speye(2^(L-2)) ⊗ X) - Y ⊗ (speye(2^(L-2)) ⊗ Y) - Z ⊗ (speye(2^(L-2)) ⊗ Z)) * 0.25 
    
    site_Lminus1 = pbc_proj * (speye(2^(L-2)) ⊗ proj(up) ⊗ speye(2)) + proj(down) ⊗ speye(2^(L-3)) ⊗ Projector |> SparseMatrixCSC{Float64, Int64}
    site_L = Projector ⊗ speye(2^(L-3)) ⊗ proj(up) + pbc_proj * (speye(2) ⊗ proj(down) ⊗ speye(2^(L-2))) |> SparseMatrixCSC{Float64, Int64}
    return vcat(projectors, [site_Lminus1, site_L])
end

function get_proj_motzkin(L::Int) ::Vector{SparseMatrixCSC{ComplexF64, Int64}} 
    projectors = [speye(3^(site-1)) ⊗ motzkin ⊗ speye(3^(L-site-1)) for site in 1:L-1]
    push!(projectors, proj(down1)⊗speye(3^(L-1)))
    push!(projectors, speye(3^(L-1))⊗proj(up1))
    return projectors
end

function get_proj_motzkin_pbc(L::Int) ::Vector{SparseMatrixCSC{ComplexF64, Int64}} 
    projectors = [speye(3^(site-1)) ⊗ motzkin ⊗ speye(3^(L-site-1)) for site in 1:L-1]
    
    P(v1,v2,v3,v4) = (v2*v4') ⊗ speye(3^(L-2)) ⊗ (v1*v3') # auto switch sites (1 and L) and use braket notation |v1v2><v3v4|
    P(v1,v2) = (v1*v1') ⊗ speye(3^(L-2)) ⊗ (v2*v2')
    
    pbc_proj = 0.5*(
        P(flat1, up1) - P(flat1, up1, up1, flat1) - P(up1, flat1, flat1, up1) + P(up1, flat1)
    +   P(flat1, down1) + P(down1, flat1) - P(flat1, down1, down1, flat1) - P(down1, flat1, flat1, down1)
    +   P(flat1,flat1) + P(down1,up1) - P(flat1,flat1,up1, down1) - P(up1, down1, flat1, flat1)
    )
    return vcat(projectors, [pbc_proj])
end

function get_proj_su2(L::Int; r::Int=1, pbc::Bool=true)::Vector{SparseMatrixCSC{ComplexF64,Int64}}
    @assert r > 0 "r must be positive"
    @assert r <= div(L,2) "r must satisfy r ≤ L/2"

    # Fast path for r == 1 reproducing original behavior
    if r == 1
        projectors = [speye(2^(site-1)) ⊗ Projector ⊗ speye(2^(L-site-1)) for site in 1:L-1]
        if pbc
            # projector between first and last qubit
            pbc_proj = (speye(2^L) - X ⊗ (speye(2^(L-2)) ⊗ X) - Y ⊗ (speye(2^(L-2)) ⊗ Y) - Z ⊗ (speye(2^(L-2)) ⊗ Z)) * 0.25
            return vcat(projectors, [pbc_proj])
        else
            return projectors
        end
    end

    projectors = Vector{SparseMatrixCSC{ComplexF64,Int64}}()

    # Helper to build a projector between sites i and j (i<j) separated by distance r
    # Using formula: 1/4 ( I - X_i X_j - Y_i Y_j - Z_i Z_j )
    function two_site_projector(i::Int, j::Int)
        # Blocks: left, site i, middle gap, site j, right
        left  = i>1       ? speye(2^(i-1))          : speye(1)
        gap   = j - i - 1 > 0 ? speye(2^(j-i-1))    : speye(1)
        right = j < L     ? speye(2^(L-j))          : speye(1)

        XiXj = left ⊗ X ⊗ gap ⊗ X ⊗ right
        YiYj = left ⊗ Y ⊗ gap ⊗ Y ⊗ right
        ZiZj = left ⊗ Z ⊗ gap ⊗ Z ⊗ right
        Id   = speye(2^L)
        return 0.25 * (Id - XiXj - YiYj - ZiZj) |> SparseMatrixCSC{ComplexF64,Int64}
    end

    pairs = Set{Tuple{Int,Int}}()
    if pbc
        for i in 1:L
            j = mod1(i + r, L)  # wrap using mod1 (same as ((i + r - 1) % L) + 1 for positive r)
            if i != j
                a,b = minmax(i,j)
                push!(pairs, (a,b))
            end
        end
    else
        for i in 1:L-r
            push!(pairs, (i, i+r))
        end
    end

    for (i,j) in sort(collect(pairs))
        push!(projectors, two_site_projector(i,j))
    end

    return projectors
end

function get_proj_aklt(L::Int64; pbc::Bool=true) ::Vector{SparseMatrixCSC{ComplexF64, Int64}}
    projectors = [speye(3^(site-1)) ⊗ Proj1 ⊗ speye(3^(L-site-1)) for site in 1:L-1]
    
    if pbc 
        # helper
        u = up1; d = down1; f = flat1
        P(v1,v2,v3,v4) = (v2*v4') ⊗ speye(3^(L-2)) ⊗ (v1*v3') # auto switch sites (1 and L) and use braket notation |v1v2><v3v4|
        P(v1,v2) = (v1*v1') ⊗ speye(3^(L-2)) ⊗ (v2*v2')

        # basis states
        phi2 = P(u,u)
        phi1 = 0.5*(P(u,f) + P(f,u) + P(u,f,f,u) + P(f,u,u,f))
        phi0 = 1/6*(P(u,d) + P(d,u) + 4*P(f,f) + P(u,d,d,u) + P(d,u,u,d) + 2*(P(u,d,f,f) + P(d,u,f,f) + P(f,f,u,d) + P(f,f,d,u)))
        phi_1= 0.5*(P(d,f) + P(f,d) + P(d,f,f,d) + P(f,d,d,f))
        phi_2= P(d,d)

        # total projector
        pbc_proj = speye(3^L) - (phi2 + phi1 + phi0 + phi_1 + phi_2)
        
        return vcat(projectors, [pbc_proj])
    end
    return projectors
end

function get_proj0_aklt(L::Int64; pbc::Bool=true) ::Vector{SparseMatrixCSC{ComplexF64, Int64}}
    proj0 = 1/sqrt(3)*(up1⊗down1 - flat1 ⊗ flat1 + down1⊗up1)
    proj0 = proj(proj0)
    projectors = [speye(3^(site-1)) ⊗ proj0 ⊗ speye(3^(L-site-1)) for site in 1:L-1]
    
    if pbc 
        # helper
        u = up1; d = down1; f = flat1
        P(v1,v2,v3,v4) = (v2*v4') ⊗ speye(3^(L-2)) ⊗ (v1*v3') # auto switch sites (1 and L) and use braket notation |v1v2><v3v4|
        P(v1,v2) = (v1*v1') ⊗ speye(3^(L-2)) ⊗ (v2*v2')

        phi0 = (P(u,d) + P(d,u) + P(f,f) + P(u,d,d,u) + P(d,u,u,d) 
            - P(u,d,f,f) - P(d,u,f,f) - P(f,f,u,d) - P(f,f,d,u))/3.0

        
        return vcat(projectors, [phi0])
    end
    return projectors
end

function get_projectors(circuit::Circuit) ::Vector{SparseMatrixCSC{ComplexF64, Int64}} 
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
    elseif trajtype == SU2PBCTrajectory
        projectors = get_proj_su2(L; pbc=true)
    elseif trajtype == SU2Trajectory
        projectors = get_proj_su2(L; pbc=false)
    elseif trajtype == AKLTPBCTrajectory
        projectors = get_proj_aklt(L)
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

    # normalize/prepare optional parameters
    noise_list = get(params, "noise", [L -> 0.0])
    noise_list = [n isa Function ? n : (_ -> n) for n in noise_list]

    meas_steps_list = map(normalize_step_function, params["meas_steps"])
    thermalization_list = map(normalize_step_function, params["thermalizationSteps"])
    meas_every_list = map(normalize_step_function, params["meas_every"])

    # create Simulation struct and fill it with Circuits
    vector_of_circuits = Vector{Circuit}(undef, 0)

    for (_,systemSize) in enumerate(params["systemSize"]),
        (_,meas_steps) in enumerate(meas_steps_list),
        (_,average) in enumerate(params["average"]),
        # (_,unitaryRate) in enumerate(params["unitaryRate"]),
        # (_,unitarySetup) in enumerate(params["unitarySetup"]),
        (_,bc) in enumerate(params["bc"]),
        (_,initialState) in enumerate(params["initialState"]),
        (_,measurement) in enumerate(params["measurement"]),
        (_,feedback) in enumerate(params["feedback"]),
        (_,noise) in enumerate(noise_list),
        (_,trajectories_averaged) in enumerate(params["trajectories_averaged"]),
        (_,thermalizationSteps) in enumerate(thermalization_list),
        (_,meas_every) in enumerate(meas_every_list),
        (_,model) in enumerate(params["model"])

        
        normalized_initial = normalize_initial_state(initialState)

        push!(vector_of_circuits, Circuit(
            systemSize,
            meas_steps(systemSize),
            average,
            # unitaryRate,
            # unitarySetup,
            bc,
            normalized_initial,
            measurement,
            feedback,
            noise(systemSize),
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

    return Simulation(vector_of_circuits, params)
end

function get_trajectories_from_simulation(sim::Simulation; state::Bool=false, projectors::Bool=false, feedbackIdx::Bool=false, traj_start::Int=1, traj_count::Int=100_000)
    trajectories = Vector{Trajectory}(undef, 0)
    for circuit in sim.params
        traj_circuit = get_trajectories_from_circuit(circuit; state=state, projectors=projectors, feedbackIdx=feedbackIdx, traj_start=traj_start, traj_count=traj_count)
        push!(trajectories,traj_circuit...)
    end
    return trajectories
end

function get_trajectories_from_circuit(circuit::Circuit; state::Bool=false, projectors::Bool=false, feedbackIdx::Bool=false, traj_start::Int=1, traj_count::Int=100_000)
    trajectories = Vector{Trajectory}(undef, 0)
    observables = get_observables(circuit)
    current_timestep = 1
    thermalized = ifelse(circuit.thermalizationSteps == 0, true, false)

    # Limit to a specific trajectory ID window for job-array splitting
    traj_end = traj_start + traj_count - 1
    
    for trajectoryID in 1:circuit.average
        if trajectoryID < traj_start || trajectoryID > traj_end
            continue
        end
        traj_type = determine_trajectory(circuit)
        traj = traj_type(  trajectoryID = trajectoryID,
                                        circuit = circuit,
                                        current_timestep = current_timestep,
                                        thermalized = thermalized,
                                        observables = observables)
        # check if trajectory has already been computed
        # First check fast lookup file (populated by prep_job), then fallback to file check
        if is_trajectory_computed(circuit, trajectoryID)
            continue
        end
        file = trajectory_to_filename(traj)
        computed_file = joinpath(traj.circuit.result_folder, "already_computed", basename(file))
        if !isfile(computed_file)
            push!(trajectories, traj)
        else
            continue
        end
        if state
            trajectories[end].state = circuit.initialState(circuit.L)
        end

        if projectors
            trajectories[end].projectors = get_projectors(circuit)
        end

        # if feedbackIdx
        #     trajectories[end].zFeedbackIndices = feedbackIndices(circuit)
        # end

    end
    return trajectories
end

function determine_trajectory(circuit::Circuit)
    traj_type = 0

    # check local spin
    if circuit.model == "fredkin"
        traj_type += 1
    elseif circuit.model == "motzkin"
        traj_type += 3
    elseif circuit.model == "su2"
        traj_type += 5
    elseif circuit.model == "aklt"
        traj_type += 7
    else
        error("Unknown model type: $(circuit.model)")
    end
    
    # check boundary condition
    if circuit.bc == :obc
        traj_type += 0
    elseif circuit.bc == :pbc
        traj_type += 1
    end

    # assert trajectory type
    if traj_type == 1
        return FredkinTrajectory
    elseif traj_type == 2
        return FredkinPBCTrajectory
    elseif traj_type == 3
        return MotzkinTrajectory
    elseif traj_type == 4
        return MotzkinPBCTrajectory
    elseif traj_type == 5
        return SU2Trajectory
    elseif traj_type == 6
        return SU2PBCTrajectory
    elseif traj_type == 7
        error("AKLT only supports PBC")
    elseif traj_type == 8
        return AKLTPBCTrajectory
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
        traj.projectors = get_projectors(traj.circuit)
    end

    if ismissing(traj.state)
        traj.state = traj.circuit.initialState(traj.circuit.L)
    end

    return 
end

function compute_missing_parameters!(traj::SpinHalfTrajectory)
    if ismissing(traj.projectors)
        traj.projectors = get_projectors(traj.circuit)
    end

    if ismissing(traj.state)
        traj.state = traj.circuit.initialState(traj.circuit.L)
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
        
        if :AncillaMutualInformation in circuit.observables
            onesDiagonal = diag(onesDiagonal ⊗ speye(4))
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