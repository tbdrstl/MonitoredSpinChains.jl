export remove_corrupted_trajectories,
average_trajectories_from_collect,
collect_data_partial,
get_computed_trajectory_ids,
get_data_folder,
is_trajectory_computed,
mark_trajectory_computed,
count_computed_trajectories,
init_lookup_cache,
cache_lookup_file,
sync_lookup_to_data,
prepare_job_array,
get_work_manifest,
save_work_manifest,
get_worker_trajectories,
sync_lookup_from_result_folder

# Get data folder from environment or use result_folder as fallback
function get_data_folder(circuit::Circuit)
    return get(ENV, "DATA", circuit.result_folder)
end

# ============================================================================
# File locking for thread-safe concurrent writes from multiple job array tasks
# ============================================================================

"""
    with_file_lock(f::Function, lockfile::String; timeout::Int=300)

Execute function `f` while holding an exclusive POSIX file lock.
Used for thread-safe writes to shared files from multiple SLURM job array tasks.
"""
function with_file_lock(f::Function, lockfile::String; timeout::Int=300)
    mkpath(dirname(lockfile))
    start_time = time()
    lock_acquired = false
    
    # Create or open lockfile
    io = open(lockfile, "w")
    file_descriptor = Base.Libc.fd(io)
    
    try
        # Try to acquire exclusive lock with timeout
        while !lock_acquired
            # LOCK_EX | LOCK_NB = 2 | 4 = 6 (exclusive, non-blocking)
            result = ccall(:flock, Cint, (Cint, Cint), file_descriptor, 6)
            if result == 0
                lock_acquired = true
            else
                if time() - start_time > timeout
                    error("Timeout waiting for file lock: $lockfile")
                end
                sleep(0.1 + rand() * 0.2)  # Random backoff to avoid thundering herd
            end
        end
        
        f()
    finally
        if lock_acquired
            ccall(:flock, Cint, (Cint, Cint), file_descriptor, 8)  # LOCK_UN = 8
        end
        close(io)
    end
end

# ============================================================================
# Lookup table for tracking computed trajectories
# ============================================================================
# 
# Design: Growable byte array (bitmap) for O(1) random access
# - Byte at position i = 0x01 if trajectory i is computed, 0x00 otherwise
# - Check: Seek to position, read 1 byte → O(1)
# - Mark done: Seek to position, write 1 byte with lock → O(1)
# - File grows automatically when circuit.average increases
#
# Note: hash(circuit) already excludes `average` (it's in VOLATILE_FIELDS),
# so circuits with same physics but different average share the same lookup file.
#
# Performance optimization for slow $DATA:
# - Copy lookup file to fast local storage at job start
# - Work with local copy during computation  
# - Sync back to $DATA at job end
# ============================================================================

# Global cache directory (set once per job via init_lookup_cache)
const LOOKUP_CACHE = Ref{Union{Nothing, String}}(nothing)

"""
    init_lookup_cache(local_dir::String=get(ENV, "TMPDIR", tempdir()))

Initialize local cache for lookup files. Call once at job start.
Copies are made from \$DATA to this fast local directory.
"""
function init_lookup_cache(local_dir::String=get(ENV, "TMPDIR", tempdir()))
    cache_dir = joinpath(local_dir, "lookup_cache")
    mkpath(cache_dir)
    LOOKUP_CACHE[] = cache_dir
    return cache_dir
end

"""
    get_lookup_cache_dir()

Get the local cache directory, or nothing if caching is disabled.
"""
function get_lookup_cache_dir()
    return LOOKUP_CACHE[]
end

"""
    get_lookup_file(circuit::Circuit, data_folder::String)

Get the path to the lookup file in permanent storage (\$DATA).
"""
function get_lookup_file(circuit::Circuit, data_folder::String)
    lookup_dir = joinpath(data_folder, "lookup")
    mkpath(lookup_dir)
    return joinpath(lookup_dir, string(hash(circuit)) * "_computed.bin")
end

"""
    get_local_lookup_file(circuit::Circuit)

Get the path to the local cached lookup file (fast storage).
Returns nothing if caching is not initialized.
"""
function get_local_lookup_file(circuit::Circuit)
    cache_dir = LOOKUP_CACHE[]
    if cache_dir === nothing
        return nothing
    end
    return joinpath(cache_dir, string(hash(circuit)) * "_computed.bin")
end

"""
    cache_lookup_file(circuit::Circuit; data_folder::String=get_data_folder(circuit))

Copy lookup file from \$DATA to local cache. Call at job start for each circuit.
Returns the local file path, or the \$DATA path if caching is disabled.
"""
function cache_lookup_file(circuit::Circuit; data_folder::String=get_data_folder(circuit))
    local_file = get_local_lookup_file(circuit)
    
    if local_file === nothing
        # Caching disabled, use $DATA directly
        return get_lookup_file(circuit, data_folder)
    end
    
    remote_file = get_lookup_file(circuit, data_folder)
    
    if isfile(remote_file) && !isfile(local_file)
        # Copy from $DATA to local cache
        cp(remote_file, local_file)
    end
    
    return local_file
end

"""
    sync_lookup_to_data(circuit::Circuit; data_folder::String=get_data_folder(circuit))

Merge local lookup file back to \$DATA. Call at job end.
Uses file locking for thread-safe merge with other job array tasks.
"""
function sync_lookup_to_data(circuit::Circuit; data_folder::String=get_data_folder(circuit))
    local_file = get_local_lookup_file(circuit)
    
    if local_file === nothing || !isfile(local_file)
        return  # Nothing to sync
    end
    
    remote_file = get_lookup_file(circuit, data_folder)
    lockfile = remote_file * ".lock"
    
    local_bytes = read(local_file)
    
    with_file_lock(lockfile) do
        if isfile(remote_file)
            remote_bytes = read(remote_file)
            # Merge: OR the bytes together (1 if either has 1)
            max_len = max(length(local_bytes), length(remote_bytes))
            merged = zeros(UInt8, max_len)
            for i in 1:length(remote_bytes)
                merged[i] = remote_bytes[i]
            end
            for i in 1:length(local_bytes)
                merged[i] |= local_bytes[i]  # OR: preserve 1s from both
            end
            write(remote_file, merged)
        else
            # No remote file, just copy local
            write(remote_file, local_bytes)
        end
    end
end

"""
    _get_active_lookup_file(circuit::Circuit, data_folder::String)

Internal: Get the lookup file to use (local cache if available, else \$DATA).
"""
function _get_active_lookup_file(circuit::Circuit, data_folder::String)
    local_file = get_local_lookup_file(circuit)
    if local_file !== nothing && isfile(local_file)
        return local_file
    end
    return get_lookup_file(circuit, data_folder)
end

"""
    _ensure_lookup_file_size(lookup_file::String, n_trajectories::Int)

Ensure the lookup file exists and is large enough for n_trajectories.
Extends the file with zeros if needed (preserves existing data).
"""
function _ensure_lookup_file_size(lookup_file::String, n_trajectories::Int)
    if !isfile(lookup_file)
        # Create new file with n_trajectories zero bytes
        open(lookup_file, "w") do io
            write(io, zeros(UInt8, n_trajectories))
        end
    else
        current_size = filesize(lookup_file)
        if current_size < n_trajectories
            # Extend file with zeros (append to existing data)
            open(lookup_file, "a") do io
                write(io, zeros(UInt8, n_trajectories - current_size))
            end
        end
    end
end

"""
    is_trajectory_computed(circuit::Circuit, trajID::Int; data_folder::String=get_data_folder(circuit))

Fast O(1) check if a specific trajectory has been computed.
Uses local cache if available, otherwise reads from \$DATA.
"""
function is_trajectory_computed(circuit::Circuit, trajID::Int; data_folder::String=get_data_folder(circuit))::Bool
    lookup_file = _get_active_lookup_file(circuit, data_folder)
    
    if !isfile(lookup_file)
        return false
    end
    
    # Bounds check against file size (not circuit.average, since file may be from older run)
    if trajID < 1 || trajID > filesize(lookup_file)
        return false
    end
    
    # Read single byte at position (trajID - 1) since Julia is 1-indexed
    open(lookup_file, "r") do io
        seek(io, trajID - 1)
        byte = read(io, UInt8)
        return byte == 0x01
    end
end

"""
    mark_trajectory_computed(circuit::Circuit, trajID::Int; data_folder::String=get_data_folder(circuit))

Mark a single trajectory as computed. O(1) operation.
Writes to local cache if available (no locking needed), otherwise to \$DATA with locking.
"""
function mark_trajectory_computed(circuit::Circuit, trajID::Int; data_folder::String=get_data_folder(circuit))
    local_file = get_local_lookup_file(circuit)
    
    if local_file !== nothing
        # Write to local cache (no locking needed - single process)
        _ensure_lookup_file_size(local_file, max(trajID, circuit.average))
        open(local_file, "r+") do io
            seek(io, trajID - 1)
            write(io, UInt8(0x01))
        end
    else
        # Write directly to $DATA with locking
        lookup_file = get_lookup_file(circuit, data_folder)
        lockfile = lookup_file * ".lock"
        
        with_file_lock(lockfile) do
            _ensure_lookup_file_size(lookup_file, max(trajID, circuit.average))
            open(lookup_file, "r+") do io
                seek(io, trajID - 1)
                write(io, UInt8(0x01))
            end
        end
    end
end

"""
    update_lookup_table(circuit::Circuit, traj_ids::Vector{Int}, data_folder::String)

Mark multiple trajectories as computed.
"""
function update_lookup_table(circuit::Circuit, traj_ids::Vector{Int}, data_folder::String)
    isempty(traj_ids) && return
    
    for trajID in traj_ids
        mark_trajectory_computed(circuit, trajID; data_folder=data_folder)
    end
end

"""
    get_computed_trajectory_ids(circuit::Circuit, data_folder::String=get_data_folder(circuit))

Get the set of all trajectory IDs that have been computed.
"""
function get_computed_trajectory_ids(circuit::Circuit, data_folder::String=get_data_folder(circuit))::Set{Int}
    lookup_file = _get_active_lookup_file(circuit, data_folder)
    
    if !isfile(lookup_file)
        return Set{Int}()
    end
    
    bytes = read(lookup_file)
    return Set{Int}(i for (i, b) in enumerate(bytes) if b == 0x01)
end

"""
    count_computed_trajectories(circuit::Circuit; data_folder::String=get_data_folder(circuit))

Fast count of how many trajectories have been computed.
"""
function count_computed_trajectories(circuit::Circuit; data_folder::String=get_data_folder(circuit))::Int
    lookup_file = _get_active_lookup_file(circuit, data_folder)
    
    if !isfile(lookup_file)
        return 0
    end
    
    bytes = read(lookup_file)
    return count(==(0x01), bytes)
end

# ============================================================================
# Prep Job: Prepare job arrays by determining work and copying lookup files
# ============================================================================
#
# Workflow:
# 1. Submit prep job (1 core, 2GB RAM, ~10 min) with dependency: none
# 2. Prep job reads $DATA lookup, determines missing trajectories
# 3. Prep job copies lookup to result_folder (fast storage)
# 4. Prep job writes "work manifest" with trajectory IDs to compute
# 5. Main jobs (dependency: afterok:prep_job) read manifest, do their chunk
# 6. Main jobs update lookup in result_folder during/after computation
# 7. Final sync job merges result_folder lookup back to $DATA
#
# Benefits:
# - Workers get pre-filtered work lists → no duplicate computation
# - Lookup on fast storage → fast checks during computation
# - Clean recovery from timeouts → just rerun, missing work is detected
# ============================================================================

"""
    get_work_manifest_file(circuit::Circuit, result_folder::String)

Get the path to the work manifest file for a circuit.
"""
function get_work_manifest_file(circuit::Circuit, result_folder::String)
    manifest_dir = joinpath(result_folder, "manifests")
    mkpath(manifest_dir)
    return joinpath(manifest_dir, string(hash(circuit)) * "_work.bin")
end

"""
    save_work_manifest(circuit::Circuit, traj_ids::Vector{Int}; result_folder::String=circuit.result_folder)

Save the list of trajectory IDs that need to be computed.
Called by prep job.
"""
function save_work_manifest(circuit::Circuit, traj_ids::Vector{Int}; result_folder::String=circuit.result_folder)
    manifest_file = get_work_manifest_file(circuit, result_folder)
    # Write as raw Int64 array for fast reading
    open(manifest_file, "w") do io
        write(io, Int64.(traj_ids))
    end
    return manifest_file
end

"""
    get_work_manifest(circuit::Circuit; result_folder::String=circuit.result_folder)

Read the list of trajectory IDs that need to be computed.
Called by worker jobs.
"""
function get_work_manifest(circuit::Circuit; result_folder::String=circuit.result_folder)::Vector{Int}
    manifest_file = get_work_manifest_file(circuit, result_folder)
    
    if !isfile(manifest_file)
        # No manifest = compute all trajectories (fallback)
        return collect(1:circuit.average)
    end
    
    bytes = read(manifest_file)
    n_ids = length(bytes) ÷ sizeof(Int64)
    if n_ids == 0
        return Int[]
    end
    
    return collect(reinterpret(Int64, bytes))
end

"""
    get_worker_trajectories(circuit::Circuit, worker_id::Int, n_workers::Int; result_folder::String=circuit.result_folder)

Get the trajectory IDs assigned to a specific worker from the work manifest.
Divides work evenly among workers.
"""
function get_worker_trajectories(circuit::Circuit, worker_id::Int, n_workers::Int; result_folder::String=circuit.result_folder)::Vector{Int}
    all_work = get_work_manifest(circuit; result_folder=result_folder)
    
    if isempty(all_work)
        return Int[]
    end
    
    # Divide work among workers (1-indexed worker_id)
    n_total = length(all_work)
    chunk_size = cld(n_total, n_workers)  # ceiling division
    
    start_idx = (worker_id - 1) * chunk_size + 1
    end_idx = min(worker_id * chunk_size, n_total)
    
    if start_idx > n_total
        return Int[]
    end
    
    return all_work[start_idx:end_idx]
end

"""
    prepare_job_array(sim::Simulation; 
                      data_folder::String=get_data_folder(sim.params[1]),
                      result_folder::String=sim.params[1].result_folder)

Prep job function: Determine which trajectories need computing and copy lookup files.

Call this from a short prep job before starting the main compute jobs.
Returns a summary Dict with statistics for each circuit.

Steps performed:
1. Read lookup from \$DATA to see what's already computed
2. Determine which trajectories (1:average) are missing
3. Copy lookup file to result_folder (fast storage)
4. Write work manifest with missing trajectory IDs

Example SLURM workflow:
```bash
# Prep job (1 core, 2GB, 10 min)
prep_job=\$(sbatch --parsable prep.sh)

# Compute jobs (wait for prep)
sbatch --dependency=afterok:\$prep_job --array=1-10 compute.sh
```
"""
function prepare_job_array(sim::Simulation; 
                           data_folder::String=get_data_folder(sim.params[1]),
                           result_folder::String=sim.params[1].result_folder)
    
    summary = Dict{UInt64, NamedTuple}()
    
    for circuit in sim.params
        # 1. Read existing lookup from $DATA
        lookup_file_data = get_lookup_file(circuit, data_folder)
        computed_ids = get_computed_trajectory_ids(circuit, data_folder)
        
        # 2. Determine missing trajectories
        all_ids = Set(1:circuit.average)
        missing_ids = sort(collect(setdiff(all_ids, computed_ids)))
        
        n_computed = length(computed_ids)
        n_missing = length(missing_ids)
        n_total = circuit.average
        
        # 3. Copy/create lookup file in result_folder (fast storage)
        lookup_dir_local = joinpath(result_folder, "lookup")
        mkpath(lookup_dir_local)
        lookup_file_local = joinpath(lookup_dir_local, string(hash(circuit)) * "_computed.bin")
        
        if isfile(lookup_file_data)
            # Copy existing lookup from $DATA
            cp(lookup_file_data, lookup_file_local; force=true)
        end
        
        # Ensure file is large enough for circuit.average
        _ensure_lookup_file_size(lookup_file_local, circuit.average)
        
        # 4. Write work manifest
        manifest_file = save_work_manifest(circuit, missing_ids; result_folder=result_folder)
        
        # Store summary
        summary[hash(circuit)] = (
            L = circuit.L,
            n_total = n_total,
            n_computed = n_computed,
            n_missing = n_missing,
            manifest_file = manifest_file,
            lookup_file = lookup_file_local
        )
        
        println("Circuit L=$(circuit.L): $n_computed/$n_total computed, $n_missing remaining")
    end
    
    return summary
end

"""
    prepare_job_array(param_file::String; data_folder::String=ENV["DATA"])

Convenience method: Load simulation from parameter file and prepare.
"""
function prepare_job_array(param_file::String; data_folder::String=get(ENV, "DATA", ""))
    params = load(param_file, "params"; iotype=IOStream)
    sim = create_simulation(params)
    return prepare_job_array(sim; data_folder=data_folder)
end

"""
    sync_lookup_from_result_folder(sim::Simulation;
                                   data_folder::String=get_data_folder(sim.params[1]),
                                   result_folder::String=sim.params[1].result_folder)

Final sync: Merge lookup files from result_folder back to \$DATA.
Call this after all compute jobs complete.
"""
function sync_lookup_from_result_folder(sim::Simulation;
                                        data_folder::String=get_data_folder(sim.params[1]),
                                        result_folder::String=sim.params[1].result_folder)
    for circuit in sim.params
        lookup_file_local = joinpath(result_folder, "lookup", string(hash(circuit)) * "_computed.bin")
        lookup_file_data = get_lookup_file(circuit, data_folder)
        lockfile = lookup_file_data * ".lock"
        
        if !isfile(lookup_file_local)
            continue
        end
        
        local_bytes = read(lookup_file_local)
        
        with_file_lock(lockfile) do
            if isfile(lookup_file_data)
                remote_bytes = read(lookup_file_data)
                # Merge: OR the bytes together
                max_len = max(length(local_bytes), length(remote_bytes))
                merged = zeros(UInt8, max_len)
                for i in 1:length(remote_bytes)
                    merged[i] = remote_bytes[i]
                end
                for i in 1:length(local_bytes)
                    merged[i] |= local_bytes[i]
                end
                write(lookup_file_data, merged)
            else
                mkpath(dirname(lookup_file_data))
                write(lookup_file_data, local_bytes)
            end
        end
        
        n_done = count(==(0x01), read(lookup_file_data))
        println("Circuit L=$(circuit.L): synced to \$DATA, $n_done/$(circuit.average) computed")
    end
end

# ============================================================================
# Partial data collection for job array tasks
# ============================================================================

"""
    get_collected_file(circuit::Circuit, data_folder::String)

Get the path to the collected data file in the DATA folder.
"""
function get_collected_file(circuit::Circuit, data_folder::String)
    collected_dir = joinpath(data_folder, "collected")
    mkpath(collected_dir)
    return joinpath(collected_dir, string(hash(circuit)) * ".jld2")
end

"""
    collect_data_partial(circuit::Circuit, traj_ids::Vector{Int}; data_folder::String=get_data_folder(circuit))

Thread-safely collect only the specified trajectories into the shared collected file.
This is designed for incremental collection from job array tasks.
"""
function collect_data_partial(circuit::Circuit, traj_ids::Vector{Int}; data_folder::String=get_data_folder(circuit))
    collected_file = get_collected_file(circuit, data_folder)
    lockfile = collected_file * ".lock"
    
    collected_ids = Int[]
    
    with_file_lock(lockfile) do
        # Read existing data first (JLD2 "a+" with IOStream is buggy on network FS)
        existing_data = Dict{String, Any}()
        if isfile(collected_file)
            try
                jldopen(collected_file, "r"; iotype=IOStream) do f
                    for key in keys(f)
                        existing_data[key] = f[key]
                    end
                end
            catch e
                @warn "Failed to read existing collected file, starting fresh: $e"
            end
        end
        
        # Collect new trajectories
        for trajID in traj_ids
            traj_filename = trajectory_hash_filename(circuit, trajID)
            file1 = joinpath(circuit.result_folder, "already_computed", traj_filename)
            if isfile(file1)
                key = string(trajID)
                if !haskey(existing_data, key)
                    try
                        observables = load(file1, "observables"; iotype=IOStream)
                        existing_data[key] = observables
                        push!(collected_ids, trajID)
                        rm(file1)
                    catch e
                        @warn "Failed to collect trajectory $trajID: $e"
                    end
                else
                    push!(collected_ids, trajID)
                    rm(file1)
                end
            end
        end
        
        # Write all data to file (overwrite)
        if !isempty(existing_data)
            jldopen(collected_file, "w"; iotype=IOStream) do f
                for (key, val) in existing_data
                    f[key] = val
                end
            end
        end
    end
    
    # Update lookup table with successfully collected trajectories
    if !isempty(collected_ids)
        update_lookup_table(circuit, collected_ids, data_folder)
    end
    
    return collected_ids
end

"""
    collect_data_partial(sim::Simulation, traj_start::Int, traj_count::Int; data_folder::String=get_data_folder(sim.params[1]))

Collect trajectories for all circuits in the simulation, for the given trajectory range.
"""
function collect_data_partial(sim::Simulation, traj_start::Int, traj_count::Int; data_folder::String=get_data_folder(sim.params[1]))
    traj_ids = collect(traj_start:min(traj_start + traj_count - 1, sim.params[1].average))
    
    for circuit in sim.params
        collect_data_partial(circuit, traj_ids; data_folder=data_folder)
    end
end

"""
    average_trajectories_from_collected(circuit::Circuit; data_folder::String=get_data_folder(circuit))

Average trajectories from the collected file in the DATA folder.
Only averages over trajectories that have actually been collected (checks lookup table).
"""
function average_trajectories_from_collected(circuit::Circuit; data_folder::String=get_data_folder(circuit))
    collected_file = get_collected_file(circuit, data_folder)
    average_dir = joinpath(data_folder, "average")
    mkpath(average_dir)
    
    if !isfile(collected_file)
        error("No collected file found: $collected_file")
    end
    
    computed_ids = get_computed_trajectory_ids(circuit, data_folder)
    if isempty(computed_ids)
        error("No trajectories have been collected yet for this circuit")
    end
    
    n_traj = length(computed_ids)
    sorted_ids = sort(collect(computed_ids))
    
    jldopen(collected_file, "r"; iotype=IOStream) do f
        # Initialize with first trajectory
        first_id = sorted_ids[1]
        observables = f[string(first_id)]
        obs2 = deepcopy(observables)
        square!(obs2)
        
        # Add remaining trajectories
        for id in sorted_ids[2:end]
            key = string(id)
            if haskey(f, key)
                obs = f[key]
                add!(observables, obs)
                square!(obs)
                add!(obs2, obs)
            end
        end
        
        divide!(observables, n_traj)
        divide!(obs2, n_traj)
        
        obstothe2 = deepcopy(observables)
        square!(obstothe2)
        
        errors = divide!(sqrt!(subtract!(obs2, obstothe2)), sqrt(n_traj))
        
        avg_file = joinpath(average_dir, string(hash(circuit)) * ".jld2")
        jldopen(avg_file, "w"; iotype=IOStream) do f
            f["observables"] = observables
            f["errors"] = errors
            f["circuit"] = circuit
            f["n_trajectories"] = n_traj
            f["trajectory_ids"] = sorted_ids
        end
    end
    
    return n_traj
end

"""
    average_trajectories_from_collected(sim::Simulation; data_folder::String=get_data_folder(sim.params[1]))

Average all circuits in the simulation from their collected files.
"""
function average_trajectories_from_collected(sim::Simulation; data_folder::String=get_data_folder(sim.params[1]))
    for circuit in sim.params
        n = average_trajectories_from_collected(circuit; data_folder=data_folder)
        println("Averaged $n trajectories for circuit with L=$(circuit.L)")
    end
end

# ============================================================================
# Original functions (kept for backwards compatibility)
# ============================================================================


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
        computed_dir = joinpath(traj.circuit.result_folder, "already_computed")
        mkpath(computed_dir)  # mkpath handles existing dirs gracefully
        new_file = joinpath(computed_dir, basename(file))
        mv(file, new_file; force=true)
    end
end

function save_traj(traj::Trajectory)
    file = trajectory_to_filename(traj)
    
    # Use IOStream instead of MmapIO to avoid Bus errors on network filesystems like WekaFS
    jldopen(file, "w"; iotype=IOStream) do f
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
            load(file; iotype=IOStream)
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
    # mark trajectory as computed in lookup file (for fast skip on future runs)
    mark_trajectory_computed(traj.circuit, traj.trajectoryID)
    traj.observables = missing
end



function trajectory_to_filename(traj::Trajectory) ::String
    filename = trajectory_hash_filename(traj.circuit, traj.trajectoryID)
    return joinpath(traj.circuit.result_folder, filename)
end

"""
    trajectory_hash_filename(circuit::Circuit, trajID::Int) -> String

Compute the filename (not full path) for a trajectory file.
This is the single source of truth for trajectory file naming.
"""
function trajectory_hash_filename(circuit::Circuit, trajID::Int) ::String
    to_hash = string(trajID) * string(hash(circuit))
    return string(hash(to_hash)) * ".jld2"
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
    file = joinpath(circuit.result_folder, "already_computed", string(hash(circuit)) * ".jld2")

    # Read existing data first (JLD2 "a+" with IOStream is buggy on network FS)
    existing_data = Dict{String, Any}()
    if isfile(file)
        try
            jldopen(file, "r"; iotype=IOStream) do f
                for key in keys(f)
                    existing_data[key] = f[key]
                end
            end
        catch e
            @warn "Failed to read existing file, starting fresh: $e"
        end
    end

    for trajID in 1:circuit.average
        traj_filename = trajectory_hash_filename(circuit, trajID)
        file1 = joinpath(circuit.result_folder, "already_computed", traj_filename)
        if !haskey(existing_data, string(trajID))
            observables = isfile(file1) ? load(file1, "observables"; iotype=IOStream) : (println(circuit); println(trajID); println(file1) ;throw(ArgumentError("No data for circuit $file1")))
            existing_data[string(trajID)] = observables
        end
    end

    jldopen(file, "w"; iotype=IOStream) do f
        for (key, val) in existing_data
            f[key] = val
        end
    end
    remove_single_trajectories(circuit)
end

function collect_data_incomplete_ramses(circuit::Circuit)
    file = joinpath(circuit.result_folder, "already_computed", string(hash(circuit)) * ".jld2")

    # Read existing data first
    existing_data = Dict{String, Any}()
    if isfile(file)
        try
            jldopen(file, "r"; iotype=IOStream) do f
                for key in keys(f)
                    existing_data[key] = f[key]
                end
            end
        catch e
            @warn "Failed to read existing file, starting fresh: $e"
        end
    end

    @showprogress for trajID in 1:circuit.average
        traj_filename = trajectory_hash_filename(circuit, trajID)
        file1 = joinpath(circuit.result_folder, "already_computed", traj_filename)
        observables = try load(file1, "observables"; iotype=IOStream); catch e; continue end
        if !haskey(existing_data, string(trajID))
            existing_data[string(trajID)] = observables
        end
    end

    jldopen(file, "w"; iotype=IOStream) do f
        for (key, val) in existing_data
            f[key] = val
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

    params_to_save = deepcopy(params)
    params_to_save["initialState"] = map(serialize_initial_state, params["initialState"])

    for key in ("meas_steps", "thermalizationSteps", "meas_every")
        if haskey(params, key)
            params_to_save[key] = map(serialize_step_function, params[key])
        end
    end

    # Use IOStream instead of MmapIO to avoid Bus errors on network filesystems like WekaFS
    jldopen(param_filename, "w"; iotype=IOStream) do f
        f["params"] = params_to_save
        f["packageVersion"] = get_package_version("MonitoredSpinChains")
    end

    return 
end

function save_parameter_file(sim::Simulation)
    save_parameter_file(sim.params_dict)
end

function average_trajectories(circuit::Circuit)

    traj_filename1 = trajectory_hash_filename(circuit, 1)
    file1 = joinpath(circuit.result_folder, "already_computed", traj_filename1)
    
    observables = isfile(file1) ? load(file1, "observables"; iotype=IOStream) : (println(circuit); println(1); println(file1) ;throw(ArgumentError("No data for circuit $file1")))
    obs2 = deepcopy(observables)
    square!(obs2)
    for trajID in 2:circuit.average
        traj_filename = trajectory_hash_filename(circuit, trajID)
        file = joinpath(circuit.result_folder, "already_computed", traj_filename)
        if isfile(file)
            obs = load(file, "observables"; iotype=IOStream)
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

    # Use IOStream instead of MmapIO for WekaFS compatibility
    jldopen(circuit_to_filename(circuit, 0; average=true), "w"; iotype=IOStream) do f
        f["observables"] = observables
        f["errors"] = errors
        f["circuit"] = circuit
    end

    return
end

# assume collect_data(circuit) has already been performend. Use the files created there to average the data
function average_trajectories_from_collect(circuit::Circuit)
    mkpath(joinpath(circuit.result_folder, "average"))
    file1 = circuit_to_filename(circuit; final=true)
    jldopen(file1, "r"; iotype=IOStream) do f
        observables = f[string(1)]
        obs2 = deepcopy(observables)
        obs2 = square!(obs2)
        for id in 2:circuit.average
            obs = f[string(id)]
            add!(observables, obs)
            square!(obs)
            add!(obs2, obs)
        end
        divide!(observables, circuit.average)
        divide!(obs2, circuit.average)

        obstothe2 = deepcopy(observables)
        square!(obstothe2)

        errors = divide!(sqrt!(subtract!(obs2, obstothe2)), sqrt(circuit.average))
        jldopen(circuit_to_filename(circuit, 0; average=true), "w"; iotype=IOStream) do ff
            ff["observables"] = observables
            ff["errors"] = errors
            ff["circuit"] = circuit
        end
    end
    return
end

function average_trajectories_from_collect_incomplete(circuit::Circuit)
    mkpath(joinpath(circuit.result_folder, "average"))
    file1 = circuit_to_filename(circuit; final=true)
    jldopen(file1, "r"; iotype=IOStream) do f
        len = length(f)
        startind = 1
        for i in 1:len
            if !haskey(f, string(i))
                continue
            end
            startind = i
        end
        observables = f[string(startind)]
        obs2 = deepcopy(observables)
        obs2 = square!(obs2)
        for id in startind+1:circuit.average
            try 
                obs = f[string(id)]
                add!(observables, obs)
                square!(obs)
                add!(obs2, obs)
            catch e
                continue
            end
        end
        divide!(observables, len)
        divide!(obs2, len)

        obstothe2 = deepcopy(observables)
        square!(obstothe2)

        errors = divide!(sqrt!(subtract!(obs2, obstothe2)), sqrt(len))
        jldopen(circuit_to_filename(circuit, 0; average=true), "w"; iotype=IOStream) do ff
            ff["observables"] = observables
            ff["errors"] = errors
            ff["circuit"] = circuit
        end
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
    mkpath(joinpath(sim.params[1].result_folder, "average"))
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
        traj_filename = trajectory_hash_filename(circ, trajID)
        file = joinpath(circ.result_folder, "already_computed", traj_filename)
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