using MonitoredSpinChains

params = Dict(
        "name" => "test",
        "systemSize" => [8],
        "meas_steps" => [x->1],
        "average" => [1],
        "bc" => [:pbc],
        "initialState" => [ranrand_spinhalf_im],
        "measurement" => [true],
        "feedback" => [:Z],
        "result_folder" => joinpath(@__DIR__, "..", "test/test_data"),
        "observables" => [:AncillaMutualInformation],
        "trajectories_averaged" => [true],
        "thermalizationSteps" => [x->1, x->0],
        "meas_every" => [x->1],
        "steadyState" => false,
        "local_spin" => [1],
    )
    mkpath(params["result_folder"])
    
    sim = create_simulation(params)

    simulate(sim)