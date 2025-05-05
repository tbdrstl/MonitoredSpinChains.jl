using MonitoredSpinChains

params = Dict(
        "name" => "test",
        "systemSize" => [6],
        "meas_steps" => [x->1],
        "average" => [10],
        "bc" => [:pbc],
        "initialState" => [rand_spinhalf_im],
        "measurement" => [true],
        "feedback" => [:Z],
        "result_folder" => joinpath(@__DIR__, "..", "test/test_data"),
        "observables" => [:OP,:EE,:M],
        "trajectories_averaged" => [true],
        "thermalizationSteps" => [x->x^3],
        "meas_every" => [x->1],
        "local_spin" => [0.5],
    )
    mkpath(params["result_folder"])
    
    sim = create_simulation(params)

    simulate(sim)