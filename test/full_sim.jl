using MonitoredSpinChains

params = Dict(
        "name" => "test",
        "systemSize" => [12],
        "meas_steps" => [x->x^3],
        "average" => [1_000],
        "bc" => [:pbc],
        "initialState" => [rand_spinhalf_im],
        "measurement" => [true],
        "feedback" => [:Z],
        "result_folder" => joinpath(@__DIR__, "..", "test/test_data"),
        "observables" => [:OP,:EEfin],
        "trajectories_averaged" => [true],
        "thermalizationSteps" => [x->0],
        "meas_every" => [x->1],
        "model" => ["su2"],
    )

    mkpath(params["result_folder"])
    
    sim = create_simulation(params)

    simulate(sim)