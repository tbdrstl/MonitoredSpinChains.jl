#!/usr/bin/env julia
# Entry point for running a simulation slice in a Slurm job array.
# Expects a JLD2 parameters file (created by save_parameter_file or user-provided) containing `params`.

using JLD2
using MonitoredSpinChains

# Simple CLI parser without extra deps
function parse_args(args)
    opts = Dict{String,Any}(
        "--params" => nothing,
        "--system-size" => nothing,
        "--traj-start" => nothing,
        "--traj-count" => nothing,
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("--params", "--system-size", "--traj-start", "--traj-count")
            if i == length(args)
                error("Missing value for $(arg)")
            end
            val = args[i+1]
            if arg in ("--traj-start", "--traj-count")
                opts[arg] = parse(Int, val)
            else
                opts[arg] = val
            end
            i += 2
        else
            error("Unknown argument $(arg)")
        end
    end
    return opts
end

function main()
    opts = parse_args(ARGS)

    params_path = something(opts["--params"], nothing)
    params_path === nothing && error("--params <path> is required")

    data = load(params_path)
    haskey(data, "params") || error("Params file must contain key 'params'")
    params = data["params"]

    # Resolve initial state identifiers (stored as strings) back to callables
    if haskey(params, "initialState")
        params["initialState"] = map(MonitoredSpinChains.normalize_initial_state, params["initialState"])
    end

    for key in ("meas_steps", "thermalizationSteps", "meas_every")
        if haskey(params, key)
            params[key] = map(MonitoredSpinChains.normalize_step_function, params[key])
        end
    end

    # Optional override: force a single system size from CLI
    if opts["--system-size"] !== nothing
        params["systemSize"] = [parse(Int, opts["--system-size"])]
    end

    full_sim = create_simulation(params)
    if isempty(full_sim.params)
        println("No circuits to run. Exiting.")
        return
    end

    simulate(full_sim; traj_start = something(opts["--traj-start"], 1), traj_count = something(opts["--traj-count"], typemax(Int)))
end

main()
