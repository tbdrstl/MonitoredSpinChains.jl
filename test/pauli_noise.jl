using MonitoredSpinChains
using Test
using LinearAlgebra
using SparseArrays
using Random
import MonitoredSpinChains: applyX!, applyY!, applyZ!, apply_pauli!, apply_pauli_noise!,
                            apply_kick!, rand_poisson, normalize_noise,
                            canonical_circuit_signature, normalize_for_hash,
                            generalized_dicke, X, Y, Z, speye, ⊗, KICK_SITE,
                            SU2PBCTrajectory, compute_missing_parameters!

# Dense single-site Pauli on the full 2^L space, big-endian (site 1 = most
# significant bit), used as the independent reference for the in-place kernels.
function dense_pauli(L::Int, site::Int, op)
    return speye(2^(site - 1)) ⊗ op ⊗ speye(2^(L - site))
end

# Singlet projector on the unordered pair (i, j), i < j, on the full 2^L space.
function dense_pair_projector(L::Int, i::Int, j::Int)
    i, j = minmax(i, j)
    P = spzeros(ComplexF64, 2^L, 2^L)
    for op in (X, Y, Z)
        P += dense_pauli(L, i, op) * dense_pauli(L, j, op)
    end
    return 0.25 * (speye(2^L) - P)
end

# W = Σ_{i<j} ⟨P̂_ij⟩, the preparation witness of Eq. (34).
function witness(psi::AbstractVector, L::Int)
    return sum(real(dot(psi, dense_pair_projector(L, i, j), psi))
               for i in 1:L-1 for j in i+1:L)
end

@testset "Pauli kernels match dense reference" begin
    L = 5
    for site in 1:L, (axis, op) in enumerate((X, Y, Z))
        psi = normalize!(randn(ComplexF64, 2^L))
        # `constants.jl` Y is -σ^y, so the reference for axis 2 picks up a sign
        # that is immaterial to every channel built from it.
        reference = dense_pauli(L, site, op) * psi
        got = apply_pauli!(copy(psi), L, site, axis)
        sign = axis == 2 ? -1 : 1
        @test got ≈ sign * reference
    end
end

@testset "Paulis are involutions" begin
    L = 4
    for site in 1:L, axis in 1:3
        psi = normalize!(randn(ComplexF64, 2^L))
        twice = apply_pauli!(apply_pauli!(copy(psi), L, site, axis), L, site, axis)
        @test twice ≈ psi
        @test norm(apply_pauli!(copy(psi), L, site, axis)) ≈ 1
    end
end

@testset "applyY! rejects a real state" begin
    @test_throws ErrorException applyY!(randn(2^3), 3, 1)
end

# The kick from the Dicke state |D_0⟩ has an exact, sampling-free answer that
# also checks Eq. (96): the three Paulis give different W, and their mean is the
# (L-1)/3 quoted in Sec. VI C for the depolarizing channel.
@testset "Random-Pauli kick on the Dicke state" begin
    for L in (4, 6, 8)
        D0 = Vector{ComplexF64}(generalized_dicke(L, 2)[div(L, 2) + 1])
        @test norm(D0) ≈ 1
        @test witness(D0, L) ≈ 0 atol = 1e-12      # |D_0⟩ is the dark state

        i0 = KICK_SITE
        expected_W = Dict(1 => (L - 2) / 4, 2 => (L - 2) / 4, 3 => L / 2)
        expected_pair = Dict(1 => (L - 2) / (4 * (L - 1)),
                             2 => (L - 2) / (4 * (L - 1)),
                             3 => L / (2 * (L - 1)))

        for axis in 1:3
            kicked = apply_pauli!(copy(D0), L, i0, axis)
            @test witness(kicked, L) ≈ expected_W[axis]

            for j in 1:L
                j == i0 && continue
                val = real(dot(kicked, dense_pair_projector(L, i0, j), kicked))
                @test val ≈ expected_pair[axis]
            end
            # pairs untouched by the kick stay dark
            for i in 1:L-1, j in i+1:L
                (i == i0 || j == i0) && continue
                @test real(dot(kicked, dense_pair_projector(L, i, j), kicked)) ≈ 0 atol = 1e-12
            end
        end

        # Channel average = (1/3) Σ_a σ^a ρ σ^a  ⟹  W = (L-1)/3, Sec. VI C.
        @test sum(expected_W[a] for a in 1:3) / 3 ≈ (L - 1) / 3
    end
end

# apply_kick! must draw the axis uniformly, so the trajectory-averaged witness
# reproduces the (1/3) Σ_a σ^a ρ σ^a channel, i.e. W = (L-1)/3.
@testset "kick averages to the depolarizing channel" begin
    Random.seed!(20240611)
    L = 6
    D0 = Vector{ComplexF64}(generalized_dicke(L, 2)[div(L, 2) + 1])
    circ = Circuit(L, 1, 1, 0.0, :noProj, :pbc, _ -> copy(D0), true, :Z, 0.0,
                   mktempdir(), [:OP], true, 0, 1, "su2", :randomPauli)

    ntraj = 4000
    acc = 0.0
    for _ in 1:ntraj
        traj = SU2PBCTrajectory(trajectoryID=1, circuit=circ,
                                current_timestep=1, thermalized=false)
        compute_missing_parameters!(traj)
        apply_kick!(traj)
        acc += witness(traj.state, L)
    end
    # per-trajectory W ∈ {1, 1, 3}, so var = 8/9 and the standard error is
    # sqrt(8/9/ntraj) ≈ 0.015; 5 σ is a comfortably non-flaky band.
    @test acc / ntraj ≈ (L - 1) / 3 atol = 0.08
end

@testset "noise respects anisotropic rates" begin
    Random.seed!(20240612)
    L = 4
    # Pure-Z noise must never leave the Sz sector: starting from a basis state,
    # every amplitude stays on that same basis index.
    circ = Circuit(L, 1, 1, 0.0, :noProj, :pbc, identity, false, :Id,
                   (0.0, 0.0, 5.0), mktempdir(), [:OP], true, 0, 1, "su2", :none)
    for _ in 1:200
        traj = SU2PBCTrajectory(trajectoryID=1, circuit=circ, current_timestep=1,
                                thermalized=true,
                                state=ComplexF64[i == 4 ? 1 : 0 for i in 1:2^L])
        traj.projectors = MonitoredSpinChains.get_projectors(circ)
        apply_pauli_noise!(traj)
        @test abs(traj.state[4]) ≈ 1        # only a phase may have changed
    end

    # Pure-X noise on a product state must move weight off that basis index.
    circX = Circuit(L, 1, 1, 0.0, :noProj, :pbc, identity, false, :Id,
                    (50.0, 0.0, 0.0), mktempdir(), [:OP], true, 0, 1, "su2", :none)
    moved = 0
    for _ in 1:200
        traj = SU2PBCTrajectory(trajectoryID=1, circuit=circX, current_timestep=1,
                                thermalized=true,
                                state=ComplexF64[i == 4 ? 1 : 0 for i in 1:2^L])
        traj.projectors = MonitoredSpinChains.get_projectors(circX)
        apply_pauli_noise!(traj)
        abs(traj.state[4]) < 0.5 && (moved += 1)
    end
    @test moved > 150
end

# total_witness goes through the collective-spin identity Eq. (56); the explicit
# Σ_{i<j} ⟨P̂_ij⟩ built from dense pair projectors is the independent reference.
@testset "total_witness matches the explicit pair sum" begin
    Random.seed!(20240613)
    for L in (4, 6, 8)
        circ = Circuit(L, 1, 1, 0.0, :noProj, :pbc, identity, true, :Z, 0.0,
                       mktempdir(), [:W], true, 0, 1, "su2", :none)
        for _ in 1:8
            psi = normalize!(randn(ComplexF64, 2^L))
            traj = SU2PBCTrajectory(trajectoryID=1, circuit=circ,
                                    current_timestep=1, thermalized=true, state=psi)
            @test total_witness(traj) ≈ witness(psi, L)
        end

        # Exact anchors: |D_0⟩ is dark, and σ^z on it is a pure J = L/2 − 1 state.
        D0 = Vector{ComplexF64}(generalized_dicke(L, 2)[div(L, 2) + 1])
        t0 = SU2PBCTrajectory(trajectoryID=1, circuit=circ, current_timestep=1,
                              thermalized=true, state=copy(D0))
        @test total_witness(t0) ≈ 0 atol = 1e-12

        for (axis, expected) in ((1, (L - 2) / 4), (2, (L - 2) / 4), (3, L / 2))
            tk = SU2PBCTrajectory(trajectoryID=1, circuit=circ, current_timestep=1,
                                  thermalized=true,
                                  state=apply_pauli!(copy(D0), L, KICK_SITE, axis))
            @test total_witness(tk) ≈ expected
        end

        # A real-valued state must work too (no complex promotion required).
        real_psi = normalize!(randn(2^L))
        tr = SU2PBCTrajectory(trajectoryID=1, circuit=circ, current_timestep=1,
                              thermalized=true, state=real_psi)
        @test total_witness(tr) ≈ witness(complex(real_psi), L)
    end
end

@testset ":W is allocated and recorded" begin
    L = 6
    params = Dict(
        "name" => "witness_test",
        "systemSize" => [L],
        "meas_steps" => [3],
        "average" => [1],
        "bc" => [:pbc],
        "initialState" => [MonitoredSpinChains.neelState],
        "measurement" => [true],
        "feedback" => [:Z],
        "result_folder" => mktempdir(),
        "observables" => [:W],
        "trajectories_averaged" => [true],
        "thermalizationSteps" => [0],
        "meas_every" => [1],
        "model" => ["su2"],
    )
    sim = create_simulation(params; testmode=true)
    circ = sim.params[1]
    obs = MonitoredSpinChains.get_observables(circ)
    @test obs.witness == zeros(3)

    traj = SU2PBCTrajectory(trajectoryID=1, circuit=circ, current_timestep=1,
                            thermalized=false)
    traj.observables = obs
    compute_missing_parameters!(traj)
    MonitoredSpinChains.time_evolve!(traj)
    @test all(isfinite, traj.observables.witness)
    @test all(>=(0), traj.observables.witness)     # Ŵ ⪰ 0, it is a sum of projectors
    # Néel is far from the symmetric manifold; W must be strictly positive there.
    @test traj.observables.witness[1] > 0
end

# meas! on SU2PBCTrajectory now dispatches to the allocation-free kernel. It must
# reproduce the sparse-projector reference exactly, feedback included. Both paths
# draw exactly one random number per measurement (feedback consumes none), so a
# shared seed forces the same measurement outcomes and the trajectories must
# agree step for step.
@testset "fast PBC measurement matches the sparse projector path" begin
    L = 8
    Pj = MonitoredSpinChains.get_proj_su2(L; pbc=true)
    # `invoke` reaches past the SU2PBCTrajectory dispatch to the generic
    # sparse-projector meas!, which is the independent reference here.
    sparse_meas!(t, s) = invoke(MonitoredSpinChains.meas!,
                                Tuple{MonitoredSpinChains.Trajectory,Int}, t, s)

    for feedback in (:Id, :Z, :Z_bond1)
        circ = Circuit(L, 1, 1, 0.0, :noProj, :pbc, identity, true, feedback, 0.0,
                       mktempdir(), [:W], true, 0, 1, "su2", :none)
        psi0 = normalize!(randn(ComplexF64, 2^L))

        Random.seed!(4242)
        ta = SU2PBCTrajectory(trajectoryID=1, circuit=circ, current_timestep=1,
                              thermalized=true, state=copy(psi0), projectors=Pj)
        sites_a = rand(1:L, 150)
        for s in sites_a
            sparse_meas!(ta, s)                      # explicit sparse reference
        end

        Random.seed!(4242)
        tb = SU2PBCTrajectory(trajectoryID=1, circuit=circ, current_timestep=1,
                              thermalized=true, state=copy(psi0), projectors=Pj)
        sites_b = rand(1:L, 150)
        for s in sites_b
            MonitoredSpinChains.meas!(tb, s)         # dispatches to the fast path
        end

        @test sites_a == sites_b
        @test norm(ta.state - tb.state) < 1e-10
        @test norm(tb.state) ≈ 1
    end
end

# Feedback must actually fire in the fast path: without it the dark state is
# never reached and W does not decay.
@testset "fast path drives the state toward the dark manifold" begin
    Random.seed!(4243)
    L = 6
    Pj = MonitoredSpinChains.get_proj_su2(L; pbc=true)
    finalW = Dict{Symbol,Float64}()
    for feedback in (:Id, :Z)
        circ = Circuit(L, 1, 1, 0.0, :noProj, :pbc, identity, true, feedback, 0.0,
                       mktempdir(), [:W], true, 0, 1, "su2", :none)
        acc = 0.0
        ntraj = 60
        for _ in 1:ntraj
            traj = SU2PBCTrajectory(trajectoryID=1, circuit=circ, current_timestep=1,
                                    thermalized=true, projectors=Pj,
                                    state=normalize!(randn(ComplexF64, 2^L)))
            for _ in 1:400
                MonitoredSpinChains.meas!(traj, rand(1:L))
            end
            acc += total_witness(traj)
        end
        finalW[feedback] = acc / ntraj
    end
    # With feedback the ensemble is absorbed into the Dicke manifold (W -> 0);
    # without it, measurement alone leaves substantial pair weight.
    @test finalW[:Z] < 0.05
    @test finalW[:Id] > 10 * max(finalW[:Z], 1e-3)
end

@testset "singlet_expectation matches the dense projector" begin
    Random.seed!(4244)
    for L in (4, 6, 8)
        for _ in 1:4
            psi = normalize!(randn(ComplexF64, 2^L))
            for i in 1:L-1, j in i+1:L
                @test MonitoredSpinChains.singlet_expectation(psi, L, i, j) ≈
                      real(dot(psi, dense_pair_projector(L, i, j), psi))
            end
        end
        # argument order must not matter, P̂_ij is symmetric
        psi = normalize!(randn(ComplexF64, 2^L))
        @test MonitoredSpinChains.singlet_expectation(psi, L, 1, 3) ≈
              MonitoredSpinChains.singlet_expectation(psi, L, 3, 1)
    end
end

# :OP / :OPH / :OPQ must be bit-for-bit what the stored-projector path produced.
@testset "projector-free :OP/:OPH/:OPQ match the sparse path" begin
    Random.seed!(4245)
    for (L, bc) in ((6, :pbc), (8, :pbc), (6, :obc), (8, :obc))
        circ = Circuit(L, 1, 1, 0.0, :noProj, bc, identity, true, :Z, 0.0,
                       mktempdir(), [:OP], true, 0, 1, "su2", :none)
        Pj = MonitoredSpinChains.get_proj_su2(L; pbc = (bc == :pbc))
        TrajT = bc == :pbc ? SU2PBCTrajectory : MonitoredSpinChains.SU2Trajectory

        for _ in 1:3
            psi = normalize!(randn(ComplexF64, 2^L))
            traj = TrajT(trajectoryID=1, circuit=circ, current_timestep=1,
                         thermalized=true, state=psi, projectors=Pj)

            # reference: the old stored-projector sum, computed here explicitly
            refOP = 0.0; refVar = 0.0
            for P in Pj
                d = real(dot(psi, P, psi)); refOP += d; refVar += d^2
            end
            nb = bc == :pbc ? L : L - 1
            refOP /= nb; refVar = refVar / nb - refOP^2

            gotOP, gotVar, gotOP2 = MonitoredSpinChains.total_projector(traj)
            @test gotOP ≈ refOP
            @test gotVar ≈ refVar atol = 1e-12
            @test gotOP2 ≈ refOP^2

            # :OPH / :OPQ against explicit dense pair projectors
            for r in (2, div(L, 2))
                (bc == :pbc && r > div(L, 2)) && continue
                pairs = if bc == :pbc
                    r == div(L, 2) && iseven(L) ?
                        [(i, i + r) for i in 1:div(L, 2)] :
                        [(i, i + r > L ? i + r - L : i + r) for i in 1:L]
                else
                    [(i, i + r) for i in 1:L-r]
                end
                vals = [real(dot(psi, dense_pair_projector(L, p[1], p[2]), psi))
                        for p in pairs]
                m = sum(vals) / length(vals)
                got = MonitoredSpinChains.projector_stats_r(traj, r)
                @test got[1] ≈ m
                @test got[2] ≈ sum(vals .^ 2) / length(vals) - m^2 atol = 1e-12
            end
        end
    end
end

@testset "PBC SU(2) runs allocate no projectors" begin
    L = 8
    circ = Circuit(L, 3, 1, 0.0, :noProj, :pbc, MonitoredSpinChains.neelState, true,
                   :Z, 0.0, mktempdir(), [:OP, :W], true, 0, 1, "su2", :none)
    @test !MonitoredSpinChains.needs_projectors(circ)

    traj = SU2PBCTrajectory(trajectoryID=1, circuit=circ, current_timestep=1,
                            thermalized=false)
    traj.observables = MonitoredSpinChains.get_observables(circ)
    compute_missing_parameters!(traj)
    @test ismissing(traj.projectors)          # never built

    MonitoredSpinChains.time_evolve!(traj)
    @test ismissing(traj.projectors)
    @test all(isfinite, traj.observables.total_proj)
    @test all(isfinite, traj.observables.witness)

    # ...but circuits that still need them must keep getting them.
    unitary = Circuit(L, 3, 1, 1.0, :su2symmetric, :pbc, MonitoredSpinChains.neelState,
                      true, :Z, 0.0, mktempdir(), [:OP], true, 0, 1, "su2", :none)
    @test MonitoredSpinChains.needs_projectors(unitary)
    obc = Circuit(L, 3, 1, 0.0, :noProj, :obc, MonitoredSpinChains.neelState, true,
                  :Z, 0.0, mktempdir(), [:OP], true, 0, 1, "su2", :none)
    @test MonitoredSpinChains.needs_projectors(obc)
    fredkin = Circuit(L, 3, 1, 0.0, :noProj, :pbc, MonitoredSpinChains.neelState, true,
                      :Z, 0.0, mktempdir(), [:OP], true, 0, 1, "fredkin", :none)
    @test MonitoredSpinChains.needs_projectors(fredkin)
end

@testset "deferred kick (kick_step)" begin
    L = 8
    D0 = Vector{ComplexF64}(generalized_dicke(L, 2)[div(L, 2) + 1])
    # No measurement and no noise: the state only ever changes when the kick
    # fires, so the recorded series pins the firing time exactly.
    for (kick_step, first_kicked_row) in ((0, 1), (1, 2), (3, 4))
        circ = Circuit(L, 6, 1, 0.0, :noProj, :pbc, _ -> copy(D0), false, :Z,
                       (0.0, 0.0, 0.0), mktempdir(), [:W], true, 0, 1, "su2",
                       :randomPauli, kick_step)
        traj = SU2PBCTrajectory(trajectoryID=1, circuit=circ, current_timestep=1,
                                thermalized=false)
        traj.observables = MonitoredSpinChains.get_observables(circ)
        compute_missing_parameters!(traj)
        MonitoredSpinChains.time_evolve!(traj)

        w = traj.observables.witness
        # rows before the kick see the untouched dark state, rows after see the
        # kicked one (W = L/2 or (L-2)/4, never 0)
        for k in 1:first_kicked_row-1
            @test w[k] ≈ 0 atol = 1e-12
        end
        for k in first_kicked_row:length(w)
            @test w[k] > 0.1
        end
        @test w[first_kicked_row] ≈ w[end]        # frozen after the kick
    end
end

@testset "kick_step validation" begin
    base = Dict(
        "name" => "kickstep", "systemSize" => [16], "meas_steps" => [10],
        "average" => [1], "bc" => [:pbc],
        "initialState" => [MonitoredSpinChains.neelState], "measurement" => [true],
        "feedback" => [:Z], "result_folder" => mktempdir(), "observables" => [:W],
        "trajectories_averaged" => [true], "thermalizationSteps" => [0],
        "meas_every" => [1], "model" => ["su2"], "kick" => [:randomPauli])

    ok = create_simulation(merge(base, Dict("kick_step" => [4])); testmode=true)
    @test ok.params[1].kick_step == 4
    @test hash(ok.params[1]) != hash(create_simulation(base; testmode=true).params[1])

    # beyond the recorded window: would never fire
    @test_throws ArgumentError create_simulation(merge(base, Dict("kick_step" => [11]));
                                                 testmode=true)
    @test_throws ArgumentError create_simulation(merge(base, Dict("kick_step" => [-1]));
                                                 testmode=true)
    # collides with a save checkpoint at L > 12 -> would double-kick on resume
    big = merge(base, Dict("meas_steps" => [100], "kick_step" => [30]))
    @test_throws ArgumentError create_simulation(big; testmode=true)
end

@testset "dickeState" begin
    for L in (4, 6, 8, 10)
        psi = MonitoredSpinChains.dickeState(L)
        @test psi isa Vector{ComplexF64}
        @test norm(psi) ≈ 1
        # identical to the generalized_dicke construction it replaces
        @test psi ≈ Vector{ComplexF64}(generalized_dicke(L, 2)[div(L, 2) + 1])
        # it is the dark state: W = 0 and every NN singlet weight vanishes
        @test witness(psi, L) ≈ 0 atol = 1e-12
        for (i, j) in MonitoredSpinChains.nn_bonds(L, :pbc)
            @test MonitoredSpinChains.singlet_expectation(psi, L, i, j) ≈ 0 atol = 1e-12
        end
    end
    @test_throws ErrorException MonitoredSpinChains.dickeState(7)
    @test MonitoredSpinChains.serialize_initial_state(MonitoredSpinChains.dickeState) ==
          "dickeState"
end

@testset "normalize_noise" begin
    @test normalize_noise(3e-3) == (1e-3, 1e-3, 1e-3)     # isotropic: total rate κ
    @test normalize_noise((0.0, 0.0, 5e-3)) == (0.0, 0.0, 5e-3)
    @test normalize_noise([1.0, 2.0, 3.0]) == (1.0, 2.0, 3.0)
    @test_throws ArgumentError normalize_noise([1.0, 2.0])
end

@testset "rand_poisson mean and support" begin
    @test rand_poisson(0.0) == 0
    @test rand_poisson(-1.0) == 0
    n = 200_000
    for λ in (0.01, 0.5, 2.0)
        s = sum(rand_poisson(λ) for _ in 1:n)
        @test s / n ≈ λ rtol = 0.05
    end
end

# The whole point of the normalize_for_hash shim: circuits computed before
# `noise` became a triple and before `kick` existed must keep their hashes, so
# data already sitting in result folders stays addressable.
@testset "noiseless circuit hashes are unchanged" begin
    circ = Circuit(8, 10, 5, 0.0, :noProj, :pbc, identity, true, :Z, 0.0,
                   "somewhere", [:OP], true, 0, 1, "su2")
    @test circ.noise == (0.0, 0.0, 0.0)
    @test circ.kick == :none

    sig = canonical_circuit_signature(circ)
    # The signature must contain the *scalar* 0.0 that old circuits carried, and
    # must not mention :kick at all.
    @test (:noise, 0.0) in sig
    @test !any(first(entry) === :kick for entry in sig)

    # Reconstruct the pre-change signature explicitly and compare hashes.
    old_fields = (:L, :meas_steps, :unitaryRate, :unitarySetup, :bc, :initialState,
                  :measurement, :feedback, :noise, :observables, :thermalizationSteps,
                  :meas_every, :model)
    old_sig = Any[]
    for fname in old_fields
        value = fname === :noise ? 0.0 : getfield(circ, fname)
        push!(old_sig, (fname, normalize_for_hash(value)))
    end
    @test hash(old_sig) == hash(sig)
    @test hash(old_sig) == hash(circ)

    # A noisy or kicked circuit must NOT collide with the noiseless one.
    noisy = Circuit(8, 10, 5, 0.0, :noProj, :pbc, identity, true, :Z,
                    (1e-3 / 3, 1e-3 / 3, 1e-3 / 3), "somewhere", [:OP], true, 0, 1,
                    "su2", :none)
    kicked = Circuit(8, 10, 5, 0.0, :noProj, :pbc, identity, true, :Z, (0.0, 0.0, 0.0),
                     "somewhere", [:OP], true, 0, 1, "su2", :randomPauli)
    @test hash(noisy) != hash(circ)
    @test hash(kicked) != hash(circ)
    @test hash(noisy) != hash(kicked)
end

@testset "create_simulation plumbing" begin
    params = Dict(
        "name" => "pauli_noise_test",
        "systemSize" => [6],
        "meas_steps" => [4],
        "average" => [1],
        "bc" => [:pbc],
        "initialState" => [MonitoredSpinChains.neelState],
        "measurement" => [true],
        "feedback" => [:Z],
        "result_folder" => mktempdir(),
        "observables" => [:OP],
        "trajectories_averaged" => [true],
        "thermalizationSteps" => [0],
        "meas_every" => [1],
        "model" => ["su2"],
        "noise" => [0.0, 3e-3, (0.0, 0.0, 1e-3)],
        "kick" => [:none, :randomPauli],
    )
    sim = create_simulation(params; testmode=true)
    @test length(sim.params) == 6
    @test Set(c.noise for c in sim.params) ==
          Set([(0.0, 0.0, 0.0), (1e-3, 1e-3, 1e-3), (0.0, 0.0, 1e-3)])
    @test Set(c.kick for c in sim.params) == Set([:none, :randomPauli])
    @test length(Set(hash(c) for c in sim.params)) == 6

    params["kick"] = [:bogus]
    @test_throws ArgumentError create_simulation(params; testmode=true)
end
