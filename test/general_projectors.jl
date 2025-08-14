using Test
using MonitoredSpinChains
using SparseArrays
using LinearAlgebra
using Statistics
import MonitoredSpinChains: get_proj_su2, X, Y, Z, 
    speye, Projector, ⊗, get_observables, get_observables!, 
    projector_stats_r, total_projector

@testset "SU2 projectors r=1 obc compatibility" begin
    L = 6
    proj_obc = get_proj_su2(L; r=1, pbc=false)
    @test length(proj_obc) == L-1
    for site in 1:L-1
        expected = speye(2^(site-1)) ⊗ Projector ⊗ speye(2^(L-site-1)) |> SparseMatrixCSC{ComplexF64,Int64}
        P = proj_obc[site]
        @test P ≈ expected
        @test P ≈ P'                      # Hermitian
        @test P * P ≈ P                   # Idempotent
    end
end

@testset "SU2 projectors r=1 pbc extra wrap projector" begin
    L = 6
    proj_pbc = get_proj_su2(L; r=1, pbc=true)
    proj_obc = get_proj_su2(L; r=1, pbc=false)
    @test length(proj_pbc) == L          # one additional projector
    for k in 1:L-1
        @test proj_pbc[k] ≈ proj_obc[k]  # first L-1 identical
    end
    # Construct expected wrap-around projector (1,L)
    IdMid = speye(2^(L-2))
    expected_wrap = (speye(2^L) - X ⊗ (IdMid ⊗ X) - Y ⊗ (IdMid ⊗ Y) - Z ⊗ (IdMid ⊗ Z)) * 0.25
    Pw = proj_pbc[end]
    @test Pw ≈ expected_wrap
    @test Pw ≈ Pw'
    @test Pw * Pw ≈ Pw
end

@testset "SU2 projectors general r pbc properties" begin
    L = 10
    for r in 1:div(L,2)
        projs = get_proj_su2(L; r=r, pbc=true)
        expected_len = (iseven(L) && r == div(L,2)) ? div(L,2) : L
        @test length(projs) == expected_len
        for P in projs
            @test P ≈ P'
            @test P * P ≈ P
        end
    end
end

@testset "SU2 projectors general r obc properties" begin
    L = 10
    for r in 1:div(L,2)
        projs = get_proj_su2(L; r=r, pbc=false)
        @test length(projs) == L - r
        for P in projs
            @test P ≈ P'
            @test P * P ≈ P
        end
    end
end

@testset "OPH/OPQ measurement non-mutating" begin
    L = 12
    params = Dict(
        "name" => "ophopq_nomutate",
        "systemSize" => [L],
        "meas_steps" => [l->1],
        "average" => [1],
        "bc" => [:pbc],
        "initialState" => [rand_spinhalf_im],
        "measurement" => [false],
        "feedback" => [:Z],
        "noise" => [0.0],
        "result_folder" => mktempdir(),
        "observables" => [:OPH,:OPQ],
        "trajectories_averaged" => [false],
        "thermalizationSteps" => [l->0],
        "meas_every" => [l->1],
        "model" => ["su2"],
    )
    sim = create_simulation(params; testmode=true)
    circuit = first(sim.params)
    traj = first(get_trajectories_from_circuit(circuit; state=true, projectors=true))
    psi_before = copy(traj.state)
    traj.observables = get_observables(circuit)
    traj.current_timestep = 1
    get_observables!(traj)
    psi_after = traj.state
    @test norm(psi_after - psi_before) < 1e-12
end

@testset "Long-range projector observables OPH/OPQ" begin
    L = 12  # divisible by 4
    params = Dict(
        "name" => "ophopqtest",
        "systemSize" => [L],
        "meas_steps" => [l->1],
        "average" => [1],
        "bc" => [:pbc],
        "initialState" => [rand_spinhalf_im],
        "measurement" => [false],
        "feedback" => [:Z],
        "result_folder" => mktempdir(),
        "observables" => [:OPQ,:OPH],
        "trajectories_averaged" => [false],
        "thermalizationSteps" => [l->0],
        "meas_every" => [l->1],
        "model" => ["su2"],
    )
    sim = create_simulation(params; testmode=true)
    circuit = first(sim.params)
    traj = first(get_trajectories_from_circuit(circuit; state=true, projectors=true))
    traj.observables = get_observables(circuit)
    traj.current_timestep = 1
    get_observables!(traj)

    psi = traj.state

    # r = L/2 (OPH)
    r_half = div(L,2)
    projs_half = get_proj_su2(L; r=r_half, pbc=true)
    exps_half = [real(dot(psi, P*psi)) for P in projs_half]
    mean_half = mean(exps_half)
    var_half = mean(exps_half .^ 2) - mean_half^2
    obs_half = traj.observables.total_proj_half[1, :]
    @test isapprox(obs_half[1], mean_half; atol=1e-10, rtol=1e-10)
    @test isapprox(obs_half[2], var_half; atol=1e-10, rtol=1e-10)
    @test isapprox(obs_half[3], obs_half[1]^2; atol=1e-12, rtol=1e-12)

    # r = L/4 (OPQ)
    r_quarter = div(L,4)
    projs_quarter = get_proj_su2(L; r=r_quarter, pbc=true)
    exps_quarter = [real(dot(psi, P*psi)) for P in projs_quarter]
    mean_quarter = mean(exps_quarter)
    var_quarter = mean(exps_quarter .^ 2) - mean_quarter^2
    obs_quarter = traj.observables.total_proj_quarter[1, :]
    @test isapprox(obs_quarter[1], mean_quarter; atol=1e-10, rtol=1e-10)
    @test isapprox(obs_quarter[2], var_quarter; atol=1e-10, rtol=1e-10)
    @test isapprox(obs_quarter[3], obs_quarter[1]^2; atol=1e-12, rtol=1e-12)
end

@testset "projector_stats_r matches explicit projectors" begin
    for L in (8, 10, 12)
        params = Dict(
            "name" => "projstats_consistency_$L",
            "systemSize" => [L],
            "meas_steps" => [l->1],
            "average" => [1],
            "bc" => [:pbc],
            "initialState" => [rand_spinhalf_im],
            "measurement" => [false],
            "feedback" => [:Z],
            "noise" => [0.0],
            "result_folder" => mktempdir(),
            "observables" => [:OP],
            "trajectories_averaged" => [false],
            "thermalizationSteps" => [l->0],
            "meas_every" => [l->1],
            "model" => ["su2"],
        )
        sim = create_simulation(params; testmode=true)
        circuit = first(sim.params)
        traj = first(get_trajectories_from_circuit(circuit; state=true, projectors=true))
        psi = traj.state
        for r in 1:div(L,2)
            iseven(L) || (r == div(L,2) && continue)  # skip invalid half-distance for odd L
            stats = projector_stats_r(traj, r)
            projs = get_proj_su2(L; r=r, pbc=true)
            exps = [real(dot(psi, P*psi)) for P in projs]
            mean_exp = mean(exps)
            var_exp = mean(exps .^ 2) - mean_exp^2
            @test isapprox(stats[1], mean_exp; atol=1e-10, rtol=1e-10)
            @test isapprox(stats[2], var_exp; atol=1e-10, rtol=1e-10)
            @test isapprox(stats[3], stats[1]^2; atol=1e-12, rtol=1e-12)
        end
    end
end

@testset "total_proj matches projector_stats_r r=1" begin
    for bc in (:obc, :pbc)
        L = 10
        params = Dict(
            "name" => "totproj_vs_stats_$bc",
            "systemSize" => [L],
            "meas_steps" => [l->1],
            "average" => [1],
            "bc" => [bc],
            "initialState" => [rand_spinhalf_im],
            "measurement" => [false],
            "feedback" => [:Z],
            "noise" => [0.0],
            "result_folder" => mktempdir(),
            "observables" => [:OP],
            "trajectories_averaged" => [false],
            "thermalizationSteps" => [l->0],
            "meas_every" => [l->1],
            "model" => ["su2"],
        )
        sim = create_simulation(params; testmode=true)
        circuit = first(sim.params)
        traj = first(get_trajectories_from_circuit(circuit; state=true, projectors=true))
        # total_projector uses stored projectors
        op_mean, op_var, op_mean2 = total_projector(traj)
        stats = projector_stats_r(traj, 1)
        @test isapprox(op_mean, stats[1]; atol=1e-10, rtol=1e-10)
        @test isapprox(op_var,  stats[2]; atol=1e-10, rtol=1e-10)
        @test isapprox(op_mean2, stats[3]; atol=1e-12, rtol=1e-12)
    end
end