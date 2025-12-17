using MonitoredSpinChains


params = Dict(
        "name" => "test",
        "systemSize" => [4,6,8],
        "meas_steps" => [x->x^3],
        "average" => [1000],
        "bc" => [:pbc,:obc],
        "initialState" => [rand_spinhalf_im, rand_spinhalf_real],
        "measurement" => [true],
        "feedback" => [:Z],
        "result_folder" => joinpath(ENV["SCRATCH"], "test"),
        "observables" => [:OP],
        "trajectories_averaged" => [true],
        "thermalizationSteps" => [x->0],
        "meas_every" => [x->1],
        "model" => ["fredkin"],
    )
    
    sim = create_simulation(params)

    simulate(sim)
