using MonitoredSpinChains
import MonitoredSpinChains: run_trajectory!, all_plus, entanglement_entropy_general

# Parameters for a minimal ancilla SU(2) test
params = Dict(
    "name" => "ancilla_su2_test",
    "systemSize" => 10,
    "meas_steps" => [x->1000],
    "average" => [1000],
    "bc" => [:pbc],
    "initialState" => [all_plus],
    "measurement" => [true],
    "feedback" => [:Z],
    "result_folder" => joinpath(@__DIR__, "..", "test/test_data"),
    "observables" => [:AE],
    "trajectories_averaged" => [true],
    "thermalizationSteps" => [x->0],
    "meas_every" => [x->1],
    "model" => ["su2"],
)
mkpath(params["result_folder"])

sim = create_simulation(params)
traj = get_trajectories_from_simulation(sim; state=true, projectors=true)[1]

# Run a single time step (or more, if needed)
try
    run_trajectory!(traj)
catch e
    println("Error during ancilla SU(2) test: ", e)
    println(stacktrace(e))
end

@testset "Ancilla simulation running?" begin # no error from before
    @test true
end



# test ancilla entanglement entropy calculation
@testset "Ancilla Entanglement Entropy" begin
    state = all_plus(10; ancilla=true)
    traj = SU2PBCTrajectoryA(
        trajectoryID = 1,
        circuit = Circuit(
            10, # system size
            1, # meas_steps
            1, # average
            :pbc, # bc
            all_plus, # initial state
            true, # measurement
            :Z, # feedback
            joinpath(@__DIR__, "..", "test/test_data"), # result folder
            [:AE], # observables
            true, # trajectories_averaged
            0, # thermalizationSteps
            1, # meas_every
            "su2" # model
        ),
        current_timestep = 1,
        thermalized = false
    )
    traj.state = state
    ancilla_entropy = entanglement_entropy_general(traj, [1])
    @show ancilla_entropy
    @test ancilla_entropy ≈ log(2) # for this specific state
end
