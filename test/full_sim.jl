using MonitoredSpinChains

params = Dict(
        "name" => "test",
        "systemSize" => [6],
        "meas_steps" => [x->x^3],
        "average" => [10],
        "bc" => [:pbc],
        "initialState" => [neelState],
        "measurement" => [true],
        "feedback" => [:Z],
        "result_folder" => joinpath(@__DIR__, "..", "test/test_data"),
        "observables" => [:M, :MX, :OP, :EE],
        "trajectories_averaged" => [true],
        "thermalizationSteps" => [x->0],
        "meas_every" => [x->1],
        "local_spin" => [0.5],
    )
    mkpath(params["result_folder"])
    
    sim = create_simulation(params)

    simulate(sim)