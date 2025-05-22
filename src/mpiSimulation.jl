function simulate(sim::Simulation)
    
    MPI.Init()


    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    world_size = MPI.Comm_size(comm)
    nworkers = world_size - 1

    root = 0
    
    rank == root && println("Using Package version ", get_package_version())
    # println("MPI successfully initialized on $(rank) of $(world_size) workers.")
    
    # collect all trajectories
    trajectories = get_trajectories_from_simulation(sim)
    ntrajectories = length(trajectories)
    MPI.Barrier(comm)

    # if simulation has been computed before (ntrajectories == 0) inform and exit
    if ntrajectories == 0

        if rank == root
            printstyled("Simulation has been computed before. Check $(normpath(sim.params_dict["result_folder"])) for results. Exiting..."; color=:reverse)
        end
        MPI.Barrier(comm)
        MPI.Finalize()
        return
    end

    MPI.Barrier(comm)
    if rank == root
        save_parameter_file(sim)
        println("Starting $(sim.name) with $(ntrajectories) trajectories on $(nworkers) workers...")
    end
    
    if world_size == 1
        @showprogress for t in trajectories
            run_trajectory!(t)
        end
        collect_data(sim)
        println("Finished $(sim.name) with $(ntrajectories) trajectories on $(nworkers) workers.")

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
    
    # make all workers collect data
    if rank == root
        println("Collecting data...")
        ncircs = 1:length(sim.params)
        part = [ncircs[i:nworkers:end] for i in 1:nworkers]
        for i in 1:nworkers
            MPI.send(part[i],comm; dest=i)
        end
    else
        todo = MPI.recv(comm)
        circs = sim.params[todo]
        for circuit in circs
            collect_data(circuit)
        end
    end

    MPI.Barrier(comm)

    # average data
    if rank == root
        println("Averaging data...")
        @time average_trajectories_from_collect(sim)
        println("Finished $(sim.name) with $(ntrajectories) trajectories on $(nworkers) workers.")
    end
    
    MPI.Barrier(comm)
    t2 = MPI.Wtime()
    if rank == root
        println("Simulation took $(t2 - t1) seconds.")
        println("Simulation took $(t2 - t1)/60 minutes.")
        println("Simulation took $(t2 - t1)/3600 hours.")
    end
    MPI.Barrier(comm)
    MPI.Finalize()
end

function simulate_many_traj(sim::Simulation; repeat::Int=100)
    
    MPI.Init()


    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    world_size = MPI.Comm_size(comm)
    nworkers = world_size - 1

    root = 0

    # println("MPI successfully initialized on $(rank) of $(world_size) workers.")
    
    for iteration in 1:repeat
        # collect all trajectories
        trajectories = get_trajectories_from_simulation(sim)
        ntrajectories = length(trajectories)

        MPI.Barrier(comm)
        if rank == root
            save_parameter_file(sim)
            println("Starting $(sim.name) with $(ntrajectories) trajectories on $(nworkers) workers...")
        end
        
        if world_size == 1
            for i in 1:ntrajectories
                run_trajectory!(trajectories[i])
            end
            collect_data_job_array(sim, iteration)
            println("Finished $(sim.name) with $(ntrajectories) trajectories on $(nworkers) workers.")

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
            for t in trajectories
                run_trajectory!(t)
            end
        end

        if rank == root
            println("Dispatched to $(nworkers) MPI procs. Waiting for results...")
        end
        MPI.Barrier(comm)
        
        if rank == root
            collect_data_job_array(sim, iteration)
            clean_after_you(sim.params)
            # println("Finished $(sim.name) with $(ntrajectories) trajectories on $(nworkers) workers.")
            println("Finished $(sim.name) with $(ntrajectories) trajectories on $(nworkers) workers, iteration $(iteration).")
        end
        
        MPI.Barrier(comm)
    end

    MPI.Finalize()
end

# store state (complexf64 -> 2*64bit float) + L sparse projectors (2^L×2^L matricies with 2^L stored, complex entries) + L Int32 z-corrections with 2^(L-1) entries each
ram_estimate_in_gb(systemSize::Int) = (128 / 8 *2^systemSize + 2^systemSize * systemSize * 128 / 8 + systemSize * 4 + 32/8*(systemSize-2)*2^(L-1)) * 1e-9

