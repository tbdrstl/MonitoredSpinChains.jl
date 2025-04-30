using MonitoredSpinChains


params = Dict(
        "name" => "test",
        "systemSize" => 4:2:20,
        "meas_steps" => [x->x^4],
        "average" => [1000],
        "bc" => [:pbc,:obc],
        "initialState" => [rand_spinhalf_im, rand_spinhalf_real],
        "measurement" => [true],
        "feedback" => [:Z],
        "result_folder" => "/scratch/doerstel/fredkin",
        "observables" => [:OP],
        "trajectories_averaged" => [true],
        "thermalizationSteps" => [x->0],
        "meas_every" => [x->1],
        "local_spin" => [0.5],
    )
    
    sim = create_simulation(params)

    simulate(sim)
