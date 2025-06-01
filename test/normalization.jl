import MonitoredSpinChains: compute_missing_parameters!, time_evolve!
import LinearAlgebra: norm

pf = Dict(
        "name" => "test",
        "systemSize" => [6],
        "meas_steps" => [x->x^3],
        "average" => [1],
        "bc" => [:pbc, :obc],
        "initialState" => [neelState],
        "measurement" => [true],
        "feedback" => [:Z],
        "result_folder" => joinpath(@__DIR__, "..", "test/test_data"),
        "observables" => [:OP,:EEfin, :M],
        "trajectories_averaged" => [true],
        "thermalizationSteps" => [x->0],
        "meas_every" => [x->1],
        "model" => ["fredkin"],
    )

pm = Dict(
    "name" => "test",
    "systemSize" => [6],
    "meas_steps" => [x->x^3],
    "average" => [1],
    "bc" => [:pbc, :obc],
    "initialState" => [flat_spin1],
    "measurement" => [true],
    "feedback" => [:Z],
    "result_folder" => joinpath(@__DIR__, "..", "test/test_data"),
    "observables" => [:OP,:EEfin, :M],
    "trajectories_averaged" => [true],
    "thermalizationSteps" => [x->0],
    "meas_every" => [x->1],
    "model" => ["motzkin"],
)

psu2 = Dict(
    "name" => "test",
    "systemSize" => [6],
    "meas_steps" => [x->x^3],
    "average" => [1],
    "bc" => [:pbc],
    "initialState" => [neelState],
    "measurement" => [true],
    "feedback" => [:Z],
    "result_folder" => joinpath(@__DIR__, "..", "test/test_data"),
    "observables" => [:OP,:EEfin, :M],
    "trajectories_averaged" => [true],
    "thermalizationSteps" => [x->0],
    "meas_every" => [x->1],
    "model" => ["fredkin"],
)

pa = Dict(
    "name" => "test",
    "systemSize" => [6],
    "meas_steps" => [x->x^3],
    "average" => [1],
    "bc" => [:pbc],
    "initialState" => [flat_spin1],
    "measurement" => [true],
    "feedback" => [:Z],
    "result_folder" => joinpath(@__DIR__, "..", "test/test_data"),
    "observables" => [:OP,:EEfin, :M],
    "trajectories_averaged" => [true],
    "thermalizationSteps" => [x->0],
    "meas_every" => [x->1],
    "model" => ["aklt"],
)

mkpath(pa["result_folder"])

@testset "Normalization perserved" begin
    for p in [pf, pm, psu2, pa]
        sim = create_simulation(p)
        traj = get_trajectories_from_simulation(sim)
        for t in traj
            compute_missing_parameters!(t)
            time_evolve!(t)

            n = (norm(t.state) ≈ 1.)
            !n && println(typeof(t))
            @test n
        end
    end
end


