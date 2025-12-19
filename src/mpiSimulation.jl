function simulate(sim::Simulation; traj_start::Int=1, traj_count::Int=100_000)
    
    MPI.Init()


    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    world_size = MPI.Comm_size(comm)
    nworkers = world_size - 1

    root = 0
    
    rank == root && println("Using Package version ", get_package_version())
    # println("MPI successfully initialized on $(rank) of $(world_size) workers.")
    
    # collect all trajectories
    trajectories = get_trajectories_from_simulation(sim; traj_start=traj_start, traj_count=traj_count)
    ntrajectories = length(trajectories)
    MPI.Barrier(comm)

    # if simulation has been computed before (ntrajectories == 0) inform and exit
    # if ntrajectories == 0

    #     if rank == root
    #         printstyled("Simulation has been computed before. Check $(normpath(sim.params_dict["result_folder"])) for results. Exiting..."; color=:reverse)
    #     end
    #     MPI.Barrier(comm)
    #     MPI.Finalize()
    #     return
    # end

    MPI.Barrier(comm)
    if rank == root
        save_parameter_file(sim)
        println("Starting simulation with $(ntrajectories) trajectories on $(nworkers) workers...")
    end
    
    if world_size == 1
        @showprogress for t in trajectories
            run_trajectory!(t)
        end
        collect_data(sim)
        average_trajectories_from_collect(sim)
        println("Finished simulation with $(ntrajectories) trajectories on $(nworkers) workers.")

        MPI.Finalize()

        return 
    end

    # make initial state function wrapper available to all workers
    if rank == root
        for i in 1:nworkers
            MPI.send(sim,comm; dest=i)
        end
    else 
        MPI.recv(comm; source=root)
    end
    MPI.Barrier(comm)

    # distribute indices to workers
    if rank == root
        # randomly distribute indices
        indices = randperm(ntrajectories)
        part = [indices[i:nworkers:end] for i in 1:nworkers]
        for i in 1:nworkers
            MPI.send(part[i],comm; dest=i)
        end
    else
        todo = MPI.recv(comm)
        trajectories = trajectories[todo]
        worker_ntrajectories = length(trajectories)

        # make one worker show its progressbar
        if rank == 1 
            @showprogress "Progress worker 1:" for t in 1:worker_ntrajectories
                run_trajectory!(trajectories[1])
                popfirst!(trajectories)
            end
        else
            for t in 1:worker_ntrajectories
                run_trajectory!(trajectories[1])
                popfirst!(trajectories)
            end
        end
    end

    if rank == root
        println("Dispatched to $(nworkers) MPI procs. Waiting for results...")
    end
    MPI.Barrier(comm)
    t1 = MPI.Wtime()
    
    # Collect data from this job array batch only (thread-safe partial collection)
    if rank == root
        println("Collecting data for trajectories $(traj_start) to $(traj_start + traj_count - 1)...")
        data_folder = get_data_folder(sim.params[1])
        traj_ids = collect(traj_start:min(traj_start + traj_count - 1, sim.params[1].average))
        
        for circuit in sim.params
            collected = collect_data_partial(circuit, traj_ids; data_folder=data_folder)
            println("Collected $(length(collected)) trajectories for L=$(circuit.L)")
        end
        
        println("Finished batch: trajectories $(traj_start)-$(traj_start + traj_count - 1)")
        println("Note: Run average_trajectories_from_collected(sim) after all batches complete.")
    end
    
    MPI.Barrier(comm)
    t2 = MPI.Wtime()
    if rank == root
        println("Collection took $(round(((t2 - t1)/3600)))h $(round((t2 - t1)/60))min $(round((t2 - t1)))s.")
        
        # Sync lookup files from result_folder back to $DATA
        data_folder = get_data_folder(sim.params[1])
        if data_folder != sim.params[1].result_folder
            println("Syncing lookup files to \$DATA...")
            sync_lookup_from_result_folder(sim; data_folder=data_folder)
        end
    end
    MPI.Barrier(comm)
    MPI.Finalize()
end

# store state (complexf64 -> 2*64bit float) + L sparse projectors (2^L×2^L matricies with 2^L stored, complex entries) + L Int32 z-corrections with 2^(L-1) entries each
ram_estimate_in_gb(systemSize::Int) = (128 / 8 *2^systemSize + 2^systemSize * systemSize * 128 / 8 + systemSize * 4 + 32/8*(systemSize-2)*2^(L-1)) * 1e-9

