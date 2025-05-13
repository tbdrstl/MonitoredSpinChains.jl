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
    
    if rank == root
        println("Collecting data...")
        @time collect_data(sim)
        println("Finished $(sim.name) with $(ntrajectories) trajectories on $(nworkers) workers.")
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

# store state (complexf64 -> 2*64bit float) + L sparse projectors (2^L×2^L matricies with 2^L stored, complex entries)
ram_estimate_in_gb(systemSize::Int) = (128 / 8 *2^systemSize + 2^systemSize * systemSize * 128 / 8) * 1e-9

