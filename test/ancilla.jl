using MonitoredSpinChains

# Parameters for a minimal ancilla SU(2) test
params = Dict(
    "name" => "ancilla_su2_test",
    "systemSize" => [10],  # small system
    "meas_steps" => [2],
    "average" => [1],
    "bc" => [:pbc],
    "initialState" => ["neelState"],
    "measurement" => [:Z],
    "feedback" => [true],
    "result_folder" => joinpath(@__DIR__, "..", "test/test_data"),
    "observables" => [:AE],  # Ancilla Mutual Information/Entropy
    "trajectories_averaged" => false,
    "thermalizationSteps" => [0],
    "meas_every" => [1],
    "model" => ["su2"]
)

sim = create_simulation(params)
traj = get_trajectories_from_simulation(sim; state=true, projectors=true)[1]

# Run a single time step (or more, if needed)
try
    run_trajectory!(traj)
    println("Ancilla entropy: ", traj.observables[:AMI])
catch e
    println("Error during ancilla SU(2) test: ", e)
    println(stacktrace(e))
end
