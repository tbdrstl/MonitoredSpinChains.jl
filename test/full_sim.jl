using MonitoredSpinChains

params = Dict(
        "name" => "test",
        "systemSize" => [12],
        "meas_steps" => [x->div(x^3,2)],
        "average" => [1000],
        "bc" => [:pbc],
        "initialState" => [neelState],
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
