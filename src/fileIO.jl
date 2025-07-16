export remove_corrupted_trajectories,
average_trajectories_from_collect


# save trajectory to file only every 10th timestep to avoid IO overhead
function save_trajectory(traj::Trajectory)
    if traj.current_timestep > traj.circuit.meas_steps*traj.circuit.meas_every
        save_traj(traj)
    end
    # do not save small trajectories inbetween. They run very fast anyways
    # if traj.circuit.L > 12 && traj.current_timestep % 30 == 0
    #     save_traj(traj)
    # end
end

function move_to_computed_folder(traj::Trajectory)
    file = trajectory_to_filename(traj)
    if isfile(file)
        # move file to already_computed folder
        new_file = joinpath(traj.circuit.result_folder, "already_computed", basename(file))
        if !isdir(joinpath(traj.circuit.result_folder, "already_computed"))
            mkdir(joinpath(traj.circuit.result_folder, "already_computed"))
        end
        mv(file, new_file)
    end
end

function save_traj(traj::Trajectory)
    file = trajectory_to_filename(traj)
    
    jldsave(file; traj.trajectoryID, traj.state, traj.observables, traj.current_timestep, circuit = traj.circuit)

    jldopen(file,"w") do f
        f["trajectoryID"] = traj.trajectoryID
        f["state"] = traj.state
        f["observables"] = traj.observables
        f["current_timestep"] = traj.current_timestep
        
        # save circuit to group to save initial state as string. Never loaded again
        circuit = JLD2.Group(f, "circuit")
        for field in fieldnames(typeof(traj.circuit))
            # Skip the `initialState` field
            if field != :initialState
                # Save each field in the "circuit" group
                circuit[string(field)] = getfield(traj.circuit, field)
            end
        end
        circuit["initialState"] = string(traj.circuit.initialState)
    end
end

function load_existing_trajectory_data!(traj::Trajectory)
    file = trajectory_to_filename(traj)

    if isfile(file)
        # remove file to ensure simulation continues (compute trajectory again)
        f = try 
            load(file)
            traj.trajectoryID = f["trajectoryID"]
            traj.state = f["state"]
            traj.observables = f["observables"]
            traj.current_timestep = f["current_timestep"]
            # if file has been saved, state is already thermalized. Important to be able to continue computation from loaded file
            traj.thermalized = true
        catch 
            rm(file)
            return
        end
    end
end

# function remove_excess_data_specialized!(traj::Union{ZFeedbackTrajectory, ZFeedbackSteadyStateTrajectory})
#     traj.zFeedbackIndices = missing
# end

# function remove_excess_data_specialized!(traj::Union{XYZFeedbackTrajectory, XYZFeedbackSteadyStateTrajectory})
#     traj.zFeedbackIndices = missing
#     traj.xFeedbackIndices = missing
# end

# julia always uses the most specialized function. Catch all other trajcetory cases here
function remove_excess_data_specialized!(traj::Trajectory)
    return
end

function remove_excess_data!(traj::Trajectory)
    traj.state = missing
    traj.projectors = missing
    remove_excess_data_specialized!(traj)
    save_trajectory(traj)
    # avoid having too many trajectories in the actively used folder
    move_to_computed_folder(traj)
    traj.observables = missing
end



function trajectory_to_filename(traj::Trajectory) ::String
    to_hash = string(traj.trajectoryID)
    to_hash *= string(hash(traj.circuit))
    filename = string(hash(to_hash)) * ".jld2"
    return joinpath(traj.circuit.result_folder, filename)
end

function circuit_to_filename(circuit::Circuit; average::Bool=false, final::Bool=false) ::String
    filename = string(hash(circuit)) * ".jld2"

    if average
        return joinpath(circuit.result_folder, "average", filename)
    elseif final
        return joinpath(circuit.result_folder, "already_computed", filename)
    else
        return joinpath(circuit.result_folder, filename)
    end
end

function circuit_to_filename(circuit::Circuit, trajID::Int64; average::Bool=false, final::Bool=false) ::String
    filename = string(hash(circuit, trajID)) * ".jld2"

    if average
        return joinpath(circuit.result_folder, "average", filename)
    elseif final
        return joinpath(circuit.result_folder, "already_computed", filename)
    else
        return joinpath(circuit.result_folder, filename)
    end
end

# function collect_data(sim::Simulation)
#     if !isdir(joinpath(sim.params[1].result_folder, "average"))
#         mkdir(joinpath(sim.params[1].result_folder, "average"))
#     end

#     for circuit in sim.params
#         collect_data(circuit)
#         if circuit.trajectories_averaged == true
#             average_trajectories(circuit)
#         end
#         # remove_single_trajectories(circuit) 
#     end
# end
function collect_data(sim::Simulation)
    circs = sim.params
    for circuit in circs
        collect_data(circuit)
    end
    return 
end


function collect_data(circuit::Circuit)
    file = joinpath(circuit.result_folder, "already_computed", basename(circuit_to_filename(circuit)))

    jldopen(file,"a+") do f
        for trajID in 1:circuit.average
            file1 = circuit_to_filename(circuit, trajID, final=true)
            observables = isfile(file1) ? load(file1, "observables") : (println(circuit); println(1); println(file1) ;throw(ArgumentError("No data for circuit $file1")))
            if !haskey(f, string(hash(circuit, trajID)))
                f[string(hash(circuit, trajID))] = observables
            end
        end
    end
    remove_single_trajectories(circuit)
end

function collect_data_incomplete_ramses(circuit::Circuit)
    file = joinpath(circuit.result_folder, "already_computed", basename(MonitoredSpinChains.circuit_to_filename(circuit)))

    jldopen(file,"a+") do f
        @showprogress for trajID in 1:circuit.average
            file1 = MonitoredSpinChains.circuit_to_filename(circuit, trajID, final=true)
            observables = try load(file1, "observables"); catch e; println(e); continue end
            if !haskey(f, string(hash(circuit, trajID)))
                f[string(hash(circuit, trajID))] = observables
            end
        end
    end
end

function save_parameter_file(params::Dict)
    if !ispath(params["result_folder"])
        mkpath(params["result_folder"])
    end

    param_filename = joinpath(params["result_folder"],"parameters.jld2")
    fileIndex = 1

    while isfile(param_filename)
        param_filename = joinpath(params["result_folder"],"parameters$(fileIndex).jld2")
        fileIndex+=1
    end

    jldsave(param_filename; params, packageVersion = get_package_version("MonitoredSpinChains"))

    return 
end

function save_parameter_file(sim::Simulation)
    save_parameter_file(sim.params_dict)
end

function average_trajectories(circuit::Circuit)

    file1 = circuit_to_filename(circuit, 1; final=true)
    
    observables = isfile(file1) ? load(file1, "observables") : (println(circuit); println(1); println(file1) ;throw(ArgumentError("No data for circuit $file1")))
    obs2 = deepcopy(observables)
    square!(obs2)
    for trajID in 2:circuit.average
        file = circuit_to_filename(circuit, trajID; final=true)
        if isfile(file)
            obs = load(file, "observables")
            add!(observables, obs)
            square!(obs)
            add!(obs2, obs)
        else
            println(circuit)
            println(trajID)
            println(file)
            throw(ArgumentError("No data for circuit $file"))
        end
    end
    divide!(observables, circuit.average)
    divide!(obs2, circuit.average)

    obstothe2 = deepcopy(observables)
    square!(obstothe2)

    errors = divide!(sqrt!(subtract!(obs2, obstothe2)), sqrt(circuit.average))

    jldsave(circuit_to_filename(circuit, 0; average=true); observables, errors, circuit)

    return
end

# assume collect_data(circuit) has already been performend. Use the files created there to average the data
function average_trajectories_from_collect(circuit::Circuit)
    if !isdir(joinpath(circuit.result_folder, "average"))
        mkdir(joinpath(circuit.result_folder, "average"))
    end
    file1 = circuit_to_filename(circuit; final=true)
    jldopen(file1,"r") do f
        observables = f[string(hash(circuit, 1))]
        obs2 = deepcopy(observables)
        obs2 = square!(obs2)
        for id in 2:circuit.average
            obs = f[string(hash(circuit, id))]
            add!(observables, obs)
            square!(obs)
            add!(obs2, obs)
        end
        divide!(observables, circuit.average)
        divide!(obs2, circuit.average)

        obstothe2 = deepcopy(observables)
        square!(obstothe2)

        errors = divide!(sqrt!(subtract!(obs2, obstothe2)), sqrt(circuit.average))
        jldsave(circuit_to_filename(circuit, 0; average=true); observables, errors, circuit)
    end
    return
end

#=
begin 
    observables = f[string(hash(circs[1],1))]
    obs2 = deepcopy(observables)
    obs2 = square!(obs2)
    @showprogress for id in 2:circs[1].average
        try
            obs = f[string(hash(circs[1], id))]
            add!(observables, obs)
            square!(obs)
            add!(obs2, obs)
            i+=1
        catch e
            continue
        end
    end
    divide!(observables, length(f))
    divide!(obs2, length(f))
    
    obstothe2 = deepcopy(observables)
    square!(obstothe2)

    errors = divide!(sqrt!(subtract!(obs2, obstothe2)), sqrt(length(f)))
end
=#

function average_trajectories_from_collect(sim::Simulation)
    if !isdir(joinpath(sim.params[1].result_folder, "average"))
        mkdir(joinpath(sim.params[1].result_folder, "average"))
    end
    for circuit in sim.params
        average_trajectories_from_collect(circuit)
    end
    return
end

function get_package_version(str::String)
    deps = dependencies()
    for (uuid, pkg) in deps
        if pkg.name == str
            return pkg.version
        end
    end
    return "Package not found"
end

function get_package_version()
    deps = dependencies()
    for (uuid, pkg) in deps
        if pkg.name == "MonitoredSpinChains"
            return pkg.version
        end
    end
    return "Package not found"
end

function remove_single_trajectories(circ::Circuit)
    for trajID in 1:circ.average
        file = circuit_to_filename(circ, trajID; final=true)
        if isfile(file)
            rm(file)
        end
    end
end

function remove_single_trajectories(sim::Simulation)
    for circuit in sim.params
        remove_single_trajectories(circuit)
    end
end



function remove_corrupted_trajectories(sim::Simulation; remove::Bool=false, verbose::Bool=true)
    traj = get_trajectories_from_simulation(sim)
    corrupted = Int[]
    @showprogress for i in 1:length(traj)
        try MonitoredSpinChains.load_existing_trajectory_data!(traj[i])
        catch e
            println(e) 
            println(i)
            if remove
                verbose && println("Removing corrupted trajectory $(i)")
                rm(trajectory_to_filename(traj[i]))
            else
                push!(corrupted, i)
                verbose && println("Corrupted trajectory $(i) found. Please remove it manually.")
            end
        end
    end
    !(remove) && println("Corrupted trajectories found: $(corrupted)")
    !(remove) && println("Recall function with kwarg remove=true to remove corrupted trajectories.")
    return corrupted
end