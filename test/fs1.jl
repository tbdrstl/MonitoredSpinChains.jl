using MonitoredSpinChains


params = Dict(
        "name" => "test",
        "systemSize" => [4],
        "meas_steps" => [x->1],
        "average" => [1000],
        "bc" => [:pbc,:obc],
        "initialState" => [rand_spinhalf_im, rand_spinhalf_real],
        "measurement" => [true],
        "feedback" => [:Z],
        "result_folder" => "/scratch/doerstel/fredkin",
        "observables" => [:OP,:EE],
        "trajectories_averaged" => [true],
        "thermalizationSteps" => [x->x^3],
        "meas_every" => [x->1],
        "model" => ["fredkin"],
    )
    
    sim = create_simulation(params)

    simulate(sim)
