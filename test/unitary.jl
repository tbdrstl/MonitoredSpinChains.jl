import MonitoredSpinChains: haar_measure, random_unitary!


@testset "Unitarity" begin
    L = 12
    proj = get_proj_su2(L)

    for unitarySetup in [:singleSpinHaar, :twoSpinHaar, :noProj]
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
            "unitaryRate" => [1.0],
            "unitarySetup" => [unitarySetup]
        )
        sim = create_simulation(params; testmode=true)
        circuit = first(sim.params)
        traj = first(get_trajectories_from_circuit(circuit; state=true, projectors=false))

        for i in 1:traj.circuit.L
            random_unitary!(traj.state, traj.circuit, i, proj)
            @test norm(traj.state) ≈ 1.0
            if !isapprox(norm(traj.state), 1.0)
                println("psi not normalized for $(unitarySetup)")
            end
        end
    end
end

@testset "Haar random matricies" begin 
    max_dim_to_test = 8
    @testset "Haar measure" begin 
        function test_haar_measure(d::Int; n::Int=100_000)
            uh = abs.(haar_measure(d)).^4
            for i in 1:n-1
                uh += abs.(haar_measure(d)).^4
            end
            return uh./n
        end

        for d in 1:max_dim_to_test
            @test all(x -> isapprox(x,2/(d*(d+1)), atol=1e-2),test_haar_measure(d))
        end
    end

    @testset "Unitarity" begin
        for d in 1:max_dim_to_test
            for i in 1:10
                uh = haar_measure(d)
                @test isapprox(uh*uh' - I(d), zeros(ComplexF64,d,d), atol=1e-10)
            end
        end
    end
end