export swap_1_L,
       swap_spin1_sites,
       swaps,
       symmetric_projector,
       collect_and_average_all,
       load_averaged_results,
       get_observable_vs_L


# ============================================================================
# Post-simulation analysis: Collect and average all circuits from params file
# ============================================================================

"""
    collect_and_average_all(params_file::String; 
                            output_file::String="",
                            data_folder::String="",
                            additional_L::Vector{Int}=Int[],
                            interactive::Bool=true)

Collect and average trajectory data for all circuits defined in a params.jld2 file.
Creates a single output file containing all averaged observables, errors, and circuit metadata.

# Arguments
- `params_file::String`: Path to the params.jld2 file used to start the simulation.
- `output_file::String`: (Optional) Path for the output file. Defaults to `<result_folder>/all_averaged.jld2`.
- `data_folder::String`: (Optional) Data folder where collected files are stored. Defaults to ENV["DATA"] or result_folder.
- `additional_L::Vector{Int}`: (Optional) Additional system sizes not in params file. These will be added to the circuits to process.
- `interactive::Bool`: (Optional, default=true) If true and `additional_L` is empty, prompts user to enter additional system sizes.

# Output Structure
The output JLD2 file contains for each circuit (keyed by its hash):
- `"<hash>/observables"`: Averaged Observables struct
- `"<hash>/errors"`: Standard errors (Observables struct)
- `"<hash>/circuit"`: The Circuit struct
- `"<hash>/n_trajectories"`: Number of trajectories averaged
- `"<hash>/L"`: System size (for convenience)

Additionally contains:
- `"params"`: The original parameters dictionary
- `"circuit_hashes"`: Vector of all circuit hashes (for iteration)
- `"summary"`: Dict with summary statistics for each circuit

# Example
```julia
# After running simulations with params.jld2 (interactive mode - will prompt for additional L)
collect_and_average_all("/path/to/params.jld2")

# With additional system sizes specified directly
collect_and_average_all("/path/to/params.jld2"; additional_L=[4, 8, 12, 16, 20])

# Non-interactive mode (no prompting)
collect_and_average_all("/path/to/params.jld2"; interactive=false)

# With custom output location
collect_and_average_all("/path/to/params.jld2"; output_file="/path/to/results.jld2")
```
"""
function collect_and_average_all(params_file::String; 
                                  output_file::String="",
                                  data_folder::String="",
                                  additional_L::Vector{Int}=Int[],
                                  interactive::Bool=true)
    # Load parameters and create simulation
    println("Loading parameters from: $params_file")
    params = load(params_file, "params"; iotype=IOStream)
    
    # Get existing system sizes from params (handle different possible key names)
    existing_L = if haskey(params, "systemSize")
        params["systemSize"]
    elseif haskey(params, "L")
        params["L"]
    elseif haskey(params, "system_size")
        params["system_size"]
    else
        # Try to find system size key
        println("Available keys in params: $(keys(params))")
        println("Could not find system size key. Please check your params file.")
        Int[]
    end
    
    # Ensure existing_L is a vector
    existing_L = existing_L isa Vector ? existing_L : [existing_L]
    println("System sizes in params file: $existing_L")
    
    # Prompt for additional system sizes if interactive mode is enabled
    if interactive && isempty(additional_L)
        println()
        println("Would you like to add additional system sizes not in the params file?")
        println("Enter comma-separated L values (e.g., 4,8,12,16,20) or press Enter to skip:")
        print("> ")
        input = readline()
        if !isempty(strip(input))
            try
                additional_L = [parse(Int, strip(s)) for s in split(input, ",")]
                # Filter out sizes already in params
                additional_L = filter(L -> !(L in existing_L), additional_L)
                if !isempty(additional_L)
                    println("Adding system sizes: $additional_L")
                end
            catch e
                println("Could not parse input, continuing with params file only: $e")
                additional_L = Int[]
            end
        end
    else
        # In non-interactive mode, filter out sizes already in params
        additional_L = filter(L -> !(L in existing_L), additional_L)
    end
    
    # Merge additional system sizes into params
    if !isempty(additional_L)
        params = deepcopy(params)
        combined_L = sort(unique(vcat(existing_L, additional_L)))
        # Update the appropriate key
        if haskey(params, "systemSize")
            params["systemSize"] = combined_L
        elseif haskey(params, "L")
            params["L"] = combined_L
        elseif haskey(params, "system_size")
            params["system_size"] = combined_L
        else
            # Default to "systemSize" which is expected by create_simulation
            params["systemSize"] = combined_L
        end
        println("Combined system sizes: $combined_L")
    end
    
    sim = create_simulation(params; testmode=true)  # testmode=true to not filter already computed
    
    # Determine data folder
    if isempty(data_folder)
        data_folder = get(ENV, "DATA", sim.params[1].result_folder)
    end
    
    # Determine output file
    if isempty(output_file)
        output_file = joinpath(sim.params[1].result_folder, "all_averaged.jld2")
    end
    mkpath(dirname(output_file))
    
    println()
    println("Data folder: $data_folder")
    println("Output file: $output_file")
    println("Number of circuits: $(length(sim.params))")
    println()
    
    # Collect results for all circuits
    results = Dict{String, Any}()
    circuit_hashes = UInt64[]
    summary = Dict{UInt64, NamedTuple}()
    
    for (idx, circuit) in enumerate(sim.params)
        circ_hash = hash(circuit)
        push!(circuit_hashes, circ_hash)
        key = string(circ_hash)
        
        print("Processing circuit $idx/$(length(sim.params)) (L=$(circuit.L))... ")
        
        try
            obs, errors, n_traj = _average_circuit_from_collected(circuit, data_folder)
            
            results["$key/observables"] = obs
            results["$key/errors"] = errors
            results["$key/circuit"] = circuit
            results["$key/n_trajectories"] = n_traj
            results["$key/L"] = circuit.L
            
            summary[circ_hash] = (
                L = circuit.L,
                n_trajectories = n_traj,
                target = circuit.average,
                complete = n_traj >= circuit.average,
                feedback = circuit.feedback,
                model = circuit.model,
                bc = circuit.bc
            )
            
            println("OK ($n_traj/$(circuit.average) trajectories)")
        catch e
            println("FAILED: $e")
            summary[circ_hash] = (
                L = circuit.L,
                n_trajectories = 0,
                target = circuit.average,
                complete = false,
                feedback = circuit.feedback,
                model = circuit.model,
                bc = circuit.bc,
                error = string(e)
            )
        end
    end
    
    # Save all results to a single file
    println()
    println("Saving results to: $output_file")
    
    jldopen(output_file, "w"; iotype=IOStream) do f
        # Save all circuit data
        for (key, val) in results
            f[key] = val
        end
        
        # Save metadata
        f["params"] = params
        f["circuit_hashes"] = circuit_hashes
        f["summary"] = summary
    end
    
    # Print summary
    println()
    println("="^60)
    println("Summary:")
    println("="^60)
    for circ_hash in circuit_hashes
        s = summary[circ_hash]
        status = s.complete ? "✓" : "✗"
        n = hasproperty(s, :n_trajectories) ? s.n_trajectories : 0
        println("  L=$(lpad(s.L, 3)) | $(s.model)/$(s.bc)/$(s.feedback) | $n/$(s.target) trajectories | $status")
    end
    println("="^60)
    
    return output_file, summary
end

"""
    _average_circuit_from_collected(circuit::Circuit, data_folder::String)

Internal function: Load and average trajectories from the collected file for a single circuit.
Returns (observables, errors, n_trajectories).
"""
function _average_circuit_from_collected(circuit::Circuit, data_folder::String)
    # Try multiple locations for collected data
    collected_file = get_collected_file(circuit, data_folder)
    
    # Fallback: check in result_folder/already_computed (old format)
    old_format_file = joinpath(circuit.result_folder, "already_computed", string(hash(circuit)) * ".jld2")
    
    # Fallback: check average folder if already computed
    avg_file = circuit_to_filename(circuit, 0; average=true)
    
    if isfile(collected_file)
        return _average_from_file(collected_file, circuit)
    elseif isfile(old_format_file)
        return _average_from_file(old_format_file, circuit)
    elseif isfile(avg_file)
        # Already averaged - load directly
        data = load(avg_file; iotype=IOStream)
        return data["observables"], data["errors"], circuit.average
    else
        error("No data found for circuit L=$(circuit.L). Checked:\n  - $collected_file\n  - $old_format_file\n  - $avg_file")
    end
end

"""
    _average_from_file(file::String, circuit::Circuit)

Internal function: Average trajectories from a JLD2 file containing individual trajectory observables.
Returns (observables, errors, n_trajectories).
"""
function _average_from_file(file::String, circuit::Circuit)
    jldopen(file, "r"; iotype=IOStream) do f
        # Find all trajectory keys (numeric keys)
        traj_keys = filter(k -> tryparse(Int, k) !== nothing, keys(f))
        
        if isempty(traj_keys)
            error("No trajectory data found in file: $file")
        end
        
        sorted_keys = sort(traj_keys, by=k -> parse(Int, k))
        n_traj = length(sorted_keys)
        
        # Initialize with first trajectory
        observables = deepcopy(f[sorted_keys[1]])
        obs2 = deepcopy(observables)
        square!(obs2)
        
        # Add remaining trajectories
        for key in sorted_keys[2:end]
            obs = f[key]
            add!(observables, obs)
            obs_sq = deepcopy(obs)
            square!(obs_sq)
            add!(obs2, obs_sq)
        end
        
        # Compute mean
        divide!(observables, n_traj)
        divide!(obs2, n_traj)
        
        # Compute standard error: sqrt((<O²> - <O>²) / N)
        obstothe2 = deepcopy(observables)
        square!(obstothe2)
        
        errors = divide!(sqrt!(subtract!(obs2, obstothe2)), sqrt(n_traj))
        
        return observables, errors, n_traj
    end
end

"""
    load_averaged_results(output_file::String)

Load the results from a file created by `collect_and_average_all`.
Returns a Dict with circuit hashes as keys and NamedTuples containing observables, errors, and circuit info.

# Example
```julia
results = load_averaged_results("all_averaged.jld2")

for (h, data) in results
    println("L=\$(data.L): EE = \$(data.observables.entanglement_entropy)")
end
```
"""
function load_averaged_results(output_file::String)
    data = load(output_file; iotype=IOStream)
    
    circuit_hashes = data["circuit_hashes"]
    results = Dict{UInt64, NamedTuple}()
    
    for h in circuit_hashes
        key = string(h)
        results[h] = (
            observables = data["$key/observables"],
            errors = data["$key/errors"],
            circuit = data["$key/circuit"],
            n_trajectories = data["$key/n_trajectories"],
            L = data["$key/L"]
        )
    end
    
    return results
end

"""
    get_observable_vs_L(output_file::String, observable::Symbol; 
                        filter_fn::Function = c -> true)

Extract a specific observable across all system sizes from an averaged results file.
Returns vectors (Ls, values, errors) suitable for plotting.

# Arguments
- `output_file::String`: Path to the output file from `collect_and_average_all`
- `observable::Symbol`: Which observable to extract (e.g., `:entanglement_entropy`, `:total_proj`)
- `filter_fn::Function`: Optional filter function that takes a Circuit and returns true/false

# Example
```julia
# Get entanglement entropy vs L for all circuits
Ls, EEs, EE_errs = get_observable_vs_L("all_averaged.jld2", :entanglement_entropy)

# Get only PBC circuits
Ls, EEs, EE_errs = get_observable_vs_L("all_averaged.jld2", :entanglement_entropy;
                                        filter_fn = c -> c.bc == :pbc)
```
"""
function get_observable_vs_L(output_file::String, observable::Symbol; 
                             filter_fn::Function = c -> true)
    results = load_averaged_results(output_file)
    
    Ls = Int[]
    values = []
    errors = []
    
    for (h, data) in results
        if !filter_fn(data.circuit)
            continue
        end
        
        obs_val = getfield(data.observables, observable)
        err_val = getfield(data.errors, observable)
        
        if !ismissing(obs_val)
            push!(Ls, data.L)
            push!(values, obs_val)
            push!(errors, err_val)
        end
    end
    
    # Sort by L
    perm = sortperm(Ls)
    return Ls[perm], values[perm], errors[perm]
end


# spin-1 swap operators 
function swap_1_L(L::Int)
    d = 3
    id = speye(d)
    # computational basis projectors |i⟩⟨j|
    basis = [sparse([i == k ? 1.0 : 0.0 for i in 1:d, j in 1:d]) for k in 1:d]

    swap = spzeros(d^L, d^L)

    for i in 1:d, j in 1:d
        op1 = basis[i] * basis[j]'  # |i⟩⟨j|
        opL = basis[j] * basis[i]'  # |j⟩⟨i|

        ops = Any[]
        push!(ops, op1)
        for _ in 2:L-1
            push!(ops, id)
        end
        push!(ops, opL)

        swap += kron(ops...)
    end

    return swap./9.
end

function swap_spin1_sites()
    S = zeros(9, 9)
    for i in 1:3, j in 1:3
        # Basis index in tensor product: |i⟩⊗|j⟩ → |j⟩⊗|i⟩
        bra = 3*(i-1) + j         # |i⟩⊗|j⟩ → row index
        ket = 3*(j-1) + i         # |j⟩⊗|i⟩ → column index
        S[bra, ket] = 1.0
    end
    return S
end

function swaps(L::Int)
    sw = swap_spin1_sites()
    swap = [speye(3^(i-1)) ⊗ sw ⊗ speye(3^(L-i-1)) for i in 1:L-1]
    return push!(swap, swap_1_L(L))
end

# documentation, that L is the system size, and d the local dimension of the Hilbert space
"""
    symmetric_projector(L::Int, d::Int)

Constructs a symmetric projector for a system of size `L` with local dimension `d`.
"""
function symmetric_projector(L::Int, d::Int)
    @assert d>0 "Local dimension `d` must be greater than 0"
    ds = generalized_dicke(L, d)
    ds = proj.(ds)
    return sum(ds)
end