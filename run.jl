using MonitoredSpinChains

run_name = "twoSpinHaar_page"
result_folder = joinpath("/scratch/fs201259/td52438", run_name)
mkpath(result_folder)

params = Dict(
    "name"                  => run_name,
    "systemSize"            => collect(4:2:20),
    "meas_steps"            => [x -> 1],
    "average"               => [100],
    "bc"                    => [:pbc],
    "initialState"          => [rand_spinhalf_im],
    "measurement"           => [false],
    "feedback"              => [:Id],
    "result_folder"         => result_folder,
    "observables"           => [:EE],
    "trajectories_averaged" => [true],
    "thermalizationSteps"   => [x -> x * x ÷ 2],
    "meas_every"            => [x -> 1],
    "model"                 => ["su2"],
    "unitaryRate"           => [1.0],
    "unitarySetup"          => [:twoSpinHaar],
)

sim = create_simulation(params)
simulate(sim)
