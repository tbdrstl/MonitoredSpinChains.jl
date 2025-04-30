using MonitoredSpinChains

params = Dict(
        "name" => "test",
        "systemSize" => [6],
        "meas_steps" => [x->x^5],
        "average" => [100],
        "bc" => [:pbc],
        "initialState" => [rand_spinhalf_im],
        "measurement" => [true],
        "feedback" => [:Z],
        "result_folder" => joinpath(@__DIR__, "..", "test/test_data"),
        "observables" => [:OP],
        "trajectories_averaged" => [true],
        "thermalizationSteps" => [x->0],
        "meas_every" => [x->1],
        "local_spin" => [0.5],
    )
    mkpath(params["result_folder"])
    
    sim = create_simulation(params)

    simulate(sim)