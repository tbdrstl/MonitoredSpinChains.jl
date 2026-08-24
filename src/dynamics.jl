function run_trajectory!(traj::Trajectory)
    traj.observables = get_observables(traj.circuit)

    load_existing_trajectory_data!(traj)
    compute_missing_parameters!(traj)
    time_evolve!(traj)
    remove_excess_data!(traj)
    # GC.gc(true)
end

function time_evolve!(traj::Trajectory)
    thermalize!(traj)
    for timestep in traj.current_timestep:traj.circuit.meas_steps*traj.circuit.meas_every
        time_step!(traj)
        get_observables!(traj)
        # Deferred kick. Fired *after* get_observables!, so the row recorded at
        # kick_step is the last pre-kick sample — set kick_step = meas_every to
        # keep row 1 as the undisturbed (thermalised) baseline and have every
        # later row show the recovery.
        if traj.circuit.kick_step == traj.current_timestep
            apply_kick!(traj)
        end
        traj.current_timestep += 1
        save_trajectory(traj)
        # GC.gc()
    end
    GC.gc(true)
end

function thermalize!(traj::Trajectory)
    if traj.thermalized
        return
    end
    for timestep in traj.current_timestep:traj.circuit.thermalizationSteps
        time_step!(traj)
        traj.current_timestep += 1
    end
    traj.current_timestep = 1

    # One-shot disturbance. With kick_step == 0 it lands here, right after
    # thermalization has reached the protocol's stationary state and before the
    # recorded window opens, so every recorded row is post-kick. Guarded by the
    # early return above: `load_existing_trajectory_data!` force-sets
    # `thermalized = true` when it resumes a saved trajectory, so a resumed run
    # never kicks twice. With thermalizationSteps == 0 the loop is empty but the
    # kick still fires. A positive kick_step defers it to `time_evolve!`.
    traj.circuit.kick_step == 0 && apply_kick!(traj)

    traj.thermalized = true
    return traj
end

function time_step!(traj::Trajectory) :: Trajectory
    unitarySteps = ceil(Int, traj.circuit.unitaryRate)
    unitaryTimeEvolProb = traj.circuit.unitaryRate / unitarySteps

    time_step!(traj.circuit, traj, unitaryTimeEvolProb, unitarySteps)
end

function time_step!(circuit::Circuit, traj::Trajectory, unitaryTimeEvolProb::Float64, unitarySteps::Int64) :: Trajectory
    @unpack L, measurement = circuit

    # Pauli noise. One time step is one measurement event, i.e. dt = 1/L in the
    # units where every bond is measured at rate γ_M = 1, so this must run once
    # per step regardless of whether measurement is enabled.
    apply_pauli_noise!(traj)

    # unitary gates
    for __ in 1:unitarySteps
        if rand() < unitaryTimeEvolProb
            random_unitary!(traj.state, circuit, rand(1:L), traj.projectors)
        end
    end

    # measurement with feedback (if enabled)
    measurement && meas!(traj, rand(1:L))
    return traj
end

"""
    do_feedback(circuit, site) -> Bool

Whether a positive (singlet) detection on the bond measured at `site` triggers
the feedback unitary. `:Z` is the draft's protocol — feedback on every measured
bond (Eq. 21-22) — and `:Z_bond1` restricts it to bond 1, which is what `:Z`
meant in earlier versions of this package.
"""
@inline function do_feedback(circuit::Circuit, site::Int)::Bool
    fb = circuit.feedback
    return fb == :Z || (fb == :Z_bond1 && site == 1)
end

function meas!(traj::Trajectory, site::Int)
    Ppsi = traj.projectors[site] * traj.state
    prob = real(dot(traj.state, Ppsi))
    if rand()<prob
        sqrtProb = sqrt(prob)
        traj.state .= Ppsi/sqrtProb

        # Positive outcome: the feedback unitary K̂_α1 = V̂_α P̂_α acts (Eq. 4).
        if do_feedback(traj.circuit, site)
            correct!(traj,site)
        end
    else
        # Null outcome: K̂_α0 = 1 − P̂_α, no feedback.
        traj.state .= (traj.state - Ppsi)/sqrt(1.0-prob)
    end

    return
end

# Specialized fast SU2 measurement dispatch.
#
# Enabled for the periodic chain only. `_meas_fast_su2_core!` reproduces the
# sparse-projector `meas!` to machine precision (verified: ‖ψ_sparse − ψ_fast‖ ~
# 4e-16 over a 200-measurement trajectory with a shared seed) while avoiding the
# two 2^L allocations that `meas!` makes per measurement.
#
# NOT enabled for SU2Trajectory (:obc). The two paths genuinely differ there:
# `get_proj_su2(L; pbc=false)` returns L-1 projectors while `time_step!` draws
# `rand(1:L)`, so the sparse path raises a BoundsError at site L whereas the fast
# path returns without measuring. Switching :obc over would silently change that
# behaviour, so it is deliberately left on the sparse path.
meas!(traj::SU2PBCTrajectory, site::Int) = _meas_fast_su2!(traj, site)

# Internal fast SU2 measurement (nearest-neighbour singlet projector)
function _meas_fast_su2!(traj::SpinHalfTrajectory, site::Int)
    L = traj.circuit.L
    pbc = (traj isa SU2PBCTrajectory)
    # determine measured pair (i,j) with i < j (bond numbering matches stored projectors order)
    if pbc && site == L
        i, j = 1, L
    else
        i = site
        j = site + 1
        if j > L
            return traj
        end
    end
    if i > j
        i, j = j, i
    end

    # Function barrier with concrete state type assumption for performance
    psi = traj.state::Vector{ComplexF64}
    singlet = _meas_fast_su2_core!(psi, L, i, j)

    # Positive outcome: the feedback unitary K̂_α1 = V̂_α P̂_α acts (Eq. 4). σ^z is
    # applied to `site`, the bond's left endpoint — the same endpoint the sparse
    # `meas!` uses, and immaterial by Eq. (22). No feedback on the null outcome.
    if singlet && do_feedback(traj.circuit, site)
        correct!(traj, site)
    end
    return traj
end

# Core, allocation-free kernel acting in-place on psi for a single nearest-neighbour
# bond (i,j). Returns `true` if the singlet outcome was drawn, so the caller can
# apply the conditional feedback unitary.
@inline function _meas_fast_su2_core!(psi::Vector{ComplexF64}, L::Int, i::Int, j::Int)
    N = length(psi)
    bit_i = L - i
    bit_j = L - j
    mask_i = UInt64(1) << bit_i
    mask_j = UInt64(1) << bit_j
    both_mask = mask_i | mask_j
    flip = both_mask

    # FIRST PASS: p = 1/2 * Σ |a01 - a10|^2 (visit pattern 01 only => (bits & both_mask) == mask_j)
    p = 0.0
    @inbounds for s_uint in UInt64(0):UInt64(N-1)
        bits = s_uint & both_mask
        if bits == mask_j  # pattern (0,1)
            t_uint = s_uint ⊻ flip         # partner (1,0)
            idx01 = Int(s_uint) + 1
            idx10 = Int(t_uint) + 1
            diff = psi[idx01] - psi[idx10]
            p += 0.5 * abs2(diff)
        end
    end
    p = clamp(p, 0.0, 1.0)

    singlet = (rand() < p)

    if singlet
        # SECOND PASS: project onto singlet (|01>-|10|)/√2; zero 00 & 11 components
        invnorm = p > 0 ? inv(sqrt(p)) : 0.0
        inv_mask_j = ~mask_j
        @inbounds for s_uint in UInt64(0):UInt64(N-1)
            bits = s_uint & both_mask
            if bits == mask_j
                t_uint  = s_uint ⊻ flip
                idx01 = Int(s_uint) + 1
                idx10 = Int(t_uint) + 1
                # derive indices for 00 & 11 within block
                s00_uint = s_uint & inv_mask_j   # clear bit_j -> 00 pattern
                t11_uint = t_uint | mask_j       # set bit_j  -> 11 pattern
                idx00 = Int(s00_uint) + 1
                idx11 = Int(t11_uint) + 1
                a01 = psi[idx01]
                a10 = psi[idx10]
                d = a01 - a10
                new01 = 0.5 * d * invnorm
                psi[idx01] = new01
                psi[idx10] = -new01
                psi[idx00] = 0.0 + 0.0im
                psi[idx11] = 0.0 + 0.0im
            end
        end
    else
        # Triplet branch: normalize remaining (1-p) subspace then symmetrize 01/10 components
        one_minus_p = 1 - p
        invnorm = one_minus_p > 0 ? inv(sqrt(one_minus_p)) : 0.0
        @inbounds @simd for k in eachindex(psi)
            psi[k] *= invnorm
        end
        @inbounds for s_uint in UInt64(0):UInt64(N-1)
            bits = s_uint & both_mask
            if bits == mask_j
                t_uint = s_uint ⊻ flip
                idx01 = Int(s_uint) + 1
                idx10 = Int(t_uint) + 1
                a01 = psi[idx01]
                a10 = psi[idx10]
                newv = 0.5 * (a01 + a10)
                psi[idx01] = newv
                psi[idx10] = newv
            end
        end
    end
    return singlet
end

function correct!(traj::SU2Trajectory, site::Int)
    controlPsiZ!(traj.state, traj.circuit.L,site)
end

function correct!(traj::SU2PBCTrajectory, site::Int)
    controlPsiZ!(traj.state, traj.circuit.L,site)
end

function correct!(traj::FredkinTrajectory, site::Int)
    L = Int(log2(length(traj.state)))
    # correct the state
    if site < L-1
        controlPsiZ!(traj.state, traj.circuit.L,site)
    elseif site == L-1
        traj.state .= (X ⊗ speye(2^(L-1))) * traj.state
    elseif site == L
        traj.state .= (speye(2^(L-1)) ⊗ X) * traj.state
    else
        error("site out of bounds")
    end
    return 
end

function correct!(traj::FredkinPBCTrajectory, site::Int)
    # correct the state
    site = mod1(site+1, traj.circuit.L)
    controlPsiZ!(traj.state, traj.circuit.L,site)

    return 
end

function correct!(traj::AKLTPBCTrajectory, site::Int)
    L = traj.circuit.L
    # correct the state
    # if rand(Bool) # correct with Z
        traj.state .= (speye(3^(site-1)) ⊗ exp_z1 ⊗ speye(3^(L-site))) * traj.state
    # else # correct with XX
    #     if site < L-1
    #         traj.state .= (speye(3^(site-1)) ⊗ Xm ⊗ Xm ⊗ speye(3^(L-site-1))) * traj.state
    #     else
    #         site = mod1(site, L)
    #         traj.state .= (Xm ⊗ speye(3^(L-2)) ⊗ Xm) * traj.state
    #     end

    # end
    # if site < L-1
    #     traj.state .= (speye(3^(site-1)) ⊗ sz ⊗ speye(3^(L-site))) * traj.state
    # else
    #     site = mod1(site, L)
    #     traj.state .= (speye(3^(site-1)) ⊗ exp_x1 ⊗ speye(3^(L-site))) * traj.state
    # end
    return 
end

function correct!(traj::MotzkinTrajectory, site::Int)
    L = traj.circuit.L
    # correct the state
    if 1 < site < L-1
        traj.state .= (speye(3^(site-1)) ⊗ exp_z1 ⊗ speye(3^(L-site))) * traj.state
    else
        site = mod1(site, L)
        traj.state .= (speye(3^(site-1)) ⊗ Px ⊗ speye(3^(L-site))) * traj.state
    end
    return 
end

function correct!(traj::MotzkinPBCTrajectory, site::Int)
    L = traj.circuit.L
    # correct the state
    traj.state .= (speye(3^(site-1)) ⊗ exp_z1 ⊗ speye(3^(L-site))) * traj.state
end

# function controlPsiZ!(state::AbstractVector{T}, feedbackIndices::Vector{I}) where {I<:Integer, T<:Union{Float64, ComplexF64}}
#     timesMinusOne!(state, feedbackIndices)
# end

function controlPsiZ!(state::AbstractVector{T}, L::Int, site::Int) where {T<:Union{Float64, ComplexF64}}
    N = length(state)

    bit_pos = L - site
    # control the state with Z1
    for i in 0:N-1
        if (i >> bit_pos) & 1 == 1
            state[i+1] *= -1.0  # Julia uses 1-based indexing
        end
    end
end

# --- Single-site Pauli gates -------------------------------------------------
#
# Bit convention follows controlPsiZ! above: site `s` occupies bit position
# `L - s` of the basis index (big-endian — site 1 is the most significant bit),
# with bit = 0 meaning |↑⟩ and bit = 1 meaning |↓⟩, so Z = diag(+1, −1).
#
# Note `constants.jl` defines `Y = [0 im; -im 0]`, i.e. −σ^y. That sign is
# immaterial for everything here: both a Pauli channel σρσ† and a kick are
# invariant under σ → −σ. These kernels use the standard σ^y = [0 -im; im 0].
#
# Each loop visits every bit-pair once, entered from its bit = 0 member (`j != i`
# holds exactly when bit `L - site` of `i` is clear), so the updates are
# alias-free without a scratch buffer.

"""
    applyX!(state, L, site)

Apply σ^x on `site` in place: a pure permutation of amplitudes.
"""
function applyX!(state::AbstractVector, L::Int, site::Int)
    N = length(state)
    mask = 1 << (L - site)
    @inbounds for i in 0:N-1
        j = i | mask
        if j != i
            state[i+1], state[j+1] = state[j+1], state[i+1]
        end
    end
    return state
end

"""
    applyY!(state, L, site)

Apply σ^y on `site` in place, using σ^y|↑⟩ = +i|↓⟩ and σ^y|↓⟩ = −i|↑⟩.
Requires a complex state vector.
"""
function applyY!(state::AbstractVector{<:Complex}, L::Int, site::Int)
    N = length(state)
    mask = 1 << (L - site)
    @inbounds for i in 0:N-1
        j = i | mask
        if j != i
            a0 = state[i+1]          # amplitude with bit = 0, i.e. |↑⟩
            a1 = state[j+1]          # amplitude with bit = 1, i.e. |↓⟩
            state[j+1] = im * a0
            state[i+1] = -im * a1
        end
    end
    return state
end

function applyY!(state::AbstractVector, L::Int, site::Int)
    error("applyY!: σ^y needs a complex state vector, got eltype $(eltype(state)). " *
          "compute_missing_parameters! promotes the state to Vector{ComplexF64} when " *
          "κ_y > 0 or a kick is set; this error means that promotion was bypassed.")
end

"""
    applyZ!(state, L, site)

Apply σ^z on `site` in place.
"""
applyZ!(state::AbstractVector, L::Int, site::Int) = (controlPsiZ!(state, L, site); state)

"""
    apply_pauli!(state, L, site, axis)

Apply σ^x (`axis = 1`), σ^y (`axis = 2`) or σ^z (`axis = 3`) on `site`, in place.
"""
function apply_pauli!(state::AbstractVector, L::Int, site::Int, axis::Int)
    axis == 1 && return applyX!(state, L, site)
    axis == 2 && return applyY!(state, L, site)
    axis == 3 && return applyZ!(state, L, site)
    error("apply_pauli!: axis must be 1 (x), 2 (y) or 3 (z), got $axis")
end

"""
    rand_poisson(λ) -> Int

Poisson deviate. Knuth's product method, which is exact and cheap at the small
rates used for the noise process; the normal approximation above λ = 30 is a
guard against a long loop and is never reached in practice.
"""
function rand_poisson(λ::Float64)::Int
    λ <= 0 && return 0
    if λ < 30.0
        threshold = exp(-λ)
        k = 0
        p = 1.0
        while true
            p *= rand()
            p <= threshold && return k
            k += 1
        end
    end
    return max(0, round(Int, λ + sqrt(λ) * randn()))
end

# Draw a Pauli axis with probability proportional to its rate.
@inline function _draw_axis(κ::NTuple{3,Float64}, κtot::Float64)::Int
    r = rand() * κtot
    r < κ[1] && return 1
    r < κ[1] + κ[2] && return 2
    return 3
end

"""
    apply_pauli_noise!(traj)

Apply the random-Pauli noise events of one time step, in place.

The channel is `𝓛 ρ = Σ_{i,a} κ_a (σ_i^a ρ σ_i^a − ρ)` with per-site rates
`κ = (κ_x, κ_y, κ_z) = circuit.noise`; isotropic noise at total rate κ per site
is `κ_a = κ/3`, which is Eq. (97) of the draft. Because every jump operator is
unitary (σ^a†σ^a = 1) there is no no-jump decay to compensate, so sampling the
jumps directly reproduces the channel exactly, with no trajectory reweighting.

One time step is one measurement event. With every bond measured at rate
γ_M = 1 on a ring of L bonds, that is dt = 1/L, and the expected number of noise
events across all L sites in a step is `L · (κ_x+κ_y+κ_z) · (1/L)`, i.e. exactly
`κ_x+κ_y+κ_z` — independent of L. The residual error from applying a whole step's
events at one point in the step rather than at their true times is O(κ/L).
"""
function apply_pauli_noise!(traj::SpinHalfTrajectory)
    κ = traj.circuit.noise
    κtot = κ[1] + κ[2] + κ[3]
    κtot <= 0 && return nothing

    L = traj.circuit.L
    for _ in 1:rand_poisson(κtot)
        apply_pauli!(traj.state, L, rand(1:L), _draw_axis(κ, κtot))
    end
    return nothing
end

function apply_pauli_noise!(traj::SpinOneTrajectory)
    all(iszero, traj.circuit.noise) || error(
        "Pauli noise is defined for spin-1/2 models only; σ^a has no meaning on the " *
        "3-dimensional local Hilbert space of model \"$(traj.circuit.model)\". " *
        "Set \"noise\" => [0.0].")
    return nothing
end

"""
    apply_kick!(traj)

Apply the one-shot disturbance of Sec. VI C, once, at the end of thermalization.

`:randomPauli` draws a ∈ {x, y, z} uniformly and applies σ^a to site
`KICK_SITE`. Averaged over trajectories this is exactly the single-site channel
`Φ[ρ] = (1/3) Σ_a σ_i^a ρ σ_i^a` — the p = 1 Pauli twirl, with no identity
branch, so an error is certain rather than merely likely. By Eq. (96) it sends
`⟨P̂_ij⟩ → (1 − ⟨P̂_ij⟩)/3` on every pair containing `i` and leaves the rest
untouched.
"""
function apply_kick!(traj::SpinHalfTrajectory)
    kick = traj.circuit.kick
    kick == :none && return nothing
    if kick == :randomPauli
        apply_pauli!(traj.state, traj.circuit.L, KICK_SITE, rand(1:3))
        return nothing
    end
    error("apply_kick!: unknown kick $kick (legal: $(legal_kicks))")
end

function apply_kick!(traj::SpinOneTrajectory)
    traj.circuit.kick == :none || error(
        "The :randomPauli kick is defined for spin-1/2 models only; σ^a has no meaning " *
        "on the 3-dimensional local Hilbert space of model \"$(traj.circuit.model)\".")
    return nothing
end

# Fast Fredkin PBC measurement (3-qubit projector) for sites 1..L-2; falls back otherwise
function meas_fast_fredkin!(traj::FredkinPBCTrajectory, site::Int)
    L = traj.circuit.L
    if 1 <= site <= L-2
        psi = traj.state::Vector{ComplexF64}
        p = _meas_fast_fredkin_core!(psi, L, site, site+1, site+2)
        # feedback logic analogous to meas!: treat success probability = p
        if rand() < p && do_feedback(traj.circuit, site)
            correct!(traj, site)
        end
        return traj
    else
        return meas!(traj, site)  # use existing projector (edge projectors)
    end
end

# Fallbacks for other trajectory types
meas_fast_fredkin!(traj::Trajectory, site::Int) = meas!(traj, site)

# Core kernel: returns probability p (post-measurement state updated in-place)
@inline function _meas_fast_fredkin_core!(psi::Vector{ComplexF64}, L::Int, i::Int, j::Int, k::Int)
    N = length(psi)
    # bit positions (MSB ordering)
    bit_i = L - i; bit_j = L - j; bit_k = L - k
    mask_i = UInt64(1) << bit_i
    mask_j = UInt64(1) << bit_j
    mask_k = UInt64(1) << bit_k
    three_mask = mask_i | mask_j | mask_k

    # FIRST PASS: accumulate p using base pattern (i,j,k) = (0,0,1)
    p = 0.0
    flip_jk = mask_j | mask_k   # 001 -> 010
    flip_j  = mask_j            # 001 -> 011

    @inbounds for s_uint in UInt64(0):UInt64(N-1)
        if (s_uint & three_mask) == mask_k
            idx001 = Int(s_uint) + 1
            idx010 = Int(s_uint ⊻ flip_jk) + 1
            idx011 = Int(s_uint ⊻ flip_j) + 1
            idx101 = Int((s_uint ⊻ flip_j) ⊻ mask_i) + 1
            diff1 = psi[idx001] - psi[idx010]
            diff2 = psi[idx011] - psi[idx101]
            p += 0.5 * (abs2(diff1) + abs2(diff2))
        end
    end
    p = clamp(p, 0.0, 1.0)

    success = rand() < p
    if success
        invnorm = p > 0 ? inv(sqrt(p)) : 0.0
        # Project: Pψ has components diff/2; then normalize => diff /(2 sqrt(p)) = diff * 0.5 * invnorm
        @inbounds for s_uint in UInt64(0):UInt64(N-1)
            if (s_uint & three_mask) == mask_k
                base = s_uint
                idx001 = Int(base) + 1
                idx010 = Int(base ⊻ flip_jk) + 1
                idx011 = Int(base ⊻ flip_j) + 1
                idx101 = Int((base ⊻ flip_j) ⊻ mask_i) + 1
                diff1 = psi[idx001] - psi[idx010]
                diff2 = psi[idx011] - psi[idx101]
                scale = 0.5 * invnorm
                new001 = diff1 * scale
                new011 = diff2 * scale
                psi[idx001] = new001
                psi[idx010] = -new001
                psi[idx011] = new011
                psi[idx101] = -new011
                # zero other four states in block: 000,100,110,111
                idx000 = Int(base ⊻ mask_k) + 1
                idx100 = Int((base ⊻ mask_k) ⊻ mask_i) + 1
                idx110 = Int(((base ⊻ mask_k) ⊻ mask_i) ⊻ mask_j) + 1
                idx111 = Int((base ⊻ mask_k) ⊻ mask_j) + 1
                psi[idx000] = 0.0 + 0.0im
                psi[idx100] = 0.0 + 0.0im
                psi[idx110] = 0.0 + 0.0im
                psi[idx111] = 0.0 + 0.0im
            end
        end
    else
        one_minus_p = 1 - p
        invnorm = one_minus_p > 0 ? inv(sqrt(one_minus_p)) : 0.0
        # Subtract projection component: delta = diff/2
        @inbounds for s_uint in UInt64(0):UInt64(N-1)
            if (s_uint & three_mask) == mask_k
                base = s_uint
                idx001 = Int(base) + 1
                idx010 = Int(base ⊻ flip_jk) + 1
                idx011 = Int(base ⊻ flip_j) + 1
                idx101 = Int((base ⊻ flip_j) ⊻ mask_i) + 1
                diff1 = psi[idx001] - psi[idx010]
                diff2 = psi[idx011] - psi[idx101]
                delta1 = diff1 * 0.5
                delta2 = diff2 * 0.5
                psi[idx001] -= delta1
                psi[idx010] += delta1
                psi[idx011] -= delta2
                psi[idx101] += delta2
            end
        end
        @inbounds @simd for k in eachindex(psi)
            psi[k] *= invnorm
        end
    end
    return p
end

# Haar random unitaries with various projection schemes

function random_unitary!(psi::AbstractVector{ComplexF64}, circuit::Circuit, site::Int64, Projectors::Vector{SparseMatrixCSC{ComplexF64, Int64}}) #:: Array{ComplexF64}
    specific_random_unitary!(psi, circuit, circuit.unitarySetup, site, Projectors)
end

specific_random_unitary!(psi::AbstractVector{ComplexF64}, circuit::Circuit, unitarySetup::Symbol, site::Int64, Projectors::Vector{SparseMatrixCSC{ComplexF64, Int64}}) = 
specific_random_unitary!(psi, circuit, Val{unitarySetup}, site, Projectors)

function specific_random_unitary!(psi::AbstractVector{ComplexF64}, circuit::Circuit, ::Type{Val{:su2symmetric}}, site::Int64, Projectors::Vector{SparseMatrixCSC{ComplexF64, Int64}})
    return random_su2_gate!(psi, circuit.L, site)
end

function specific_random_unitary!(psi::AbstractVector{ComplexF64}, circuit::Circuit, ::Type{Val{:singleSpinHaar}}, site::Int64, Projectors::Vector{SparseMatrixCSC{ComplexF64, Int64}})
    if site  == circuit.L
        Ppsi = Projectors[1] * psi
        if :AncillaMutualInformation in circuit.observables
            psi .= (speye(2^(site - 1)) ⊗ haar_measure(2) ⊗ speye(2^(circuit.L - site + 2))) * Ppsi + psi - Ppsi
        else
            psi .= (speye(2^(site - 1)) ⊗ haar_measure(2) ⊗ speye(2^(circuit.L - site))) * Ppsi + psi - Ppsi
        end
    else 
        Ppsi = Projectors[site+1] * psi
        if :AncillaMutualInformation in circuit.observables
            psi .= (speye(2^(site - 1)) ⊗ haar_measure(2) ⊗ speye(2^(circuit.L - site + 2))) * Ppsi + psi - Ppsi
        else
            psi .= (speye(2^(site - 1)) ⊗ haar_measure(2) ⊗ speye(2^(circuit.L - site))) * Ppsi + psi - Ppsi
        end
    end
    # return psi
end

function specific_random_unitary!(psi::AbstractVector{ComplexF64}, circuit::Circuit, ::Type{Val{:twoSpinHaar}}, site::Int64, Projectors::Vector{SparseMatrixCSC{ComplexF64, Int64}})
    circuit.L < 4 && return
    circuit.bc == :obc && site > circuit.L - 3 && return

    Ppsi = Vector{ComplexF64}(undef, 2^circuit.L)
    if site == circuit.L
        Ppsi .= Projectors[1] * psi
    else
        Ppsi .= Projectors[site+1] * psi
    end

    if :AncillaMutualInformation in circuit.observables
        psi .= (random_haar(site,circuit.L)⊗speye(4)) * Ppsi + psi - Ppsi
    else
        psi .= random_haar(site,circuit.L) * Ppsi + psi - Ppsi
    end
    # return psi
end

function specific_random_unitary!(psi::AbstractVector{ComplexF64}, circuit::Circuit, ::Type{Val{:noProj}}, site::Int64, Projectors::Vector{SparseMatrixCSC{ComplexF64, Int64}})
    if :AncillaMutualInformation in circuit.observables
        psi .= (random_haar(site,circuit.L) ⊗ speye(4)) * psi
    else
        psi .= random_haar(site,circuit.L) * psi
    end
    # return psi
end

function random_haar(site::Int,L::Int) :: AbstractMatrix{ComplexF64}
    U4 = haar_measure(4)
    
    # n = size(U4, 1) ÷ 2
    @views quadrants = (
        U4[1:2, 1:2],       # Top Left
        U4[3:end, 1:2],   # Bottom Left
        U4[1:2, 3:end],   # Top Right
        U4[3:end, 3:end] # Bottom Right
    )
    
    if site+3<=L 
        return random_haar_bulk(site,L,quadrants)
    elseif site+3==L+1
        return random_haar_UB_other_side(L, quadrants)
    elseif site+3 == L+2
        return random_haar_proj_splitted(L, quadrants)
    else #site+3 == L+3 
        return random_haar_proj_other_side(L, quadrants)
    end
end

function just_random_haar_splitted(L::Int)
    U4 = haar_measure(4)
    
    # n = size(U4, 1) ÷ 2
    @views quadrants = (
        U4[1:2, 1:2],       # Top Left
        U4[3:end, 1:2],   # Bottom Left
        U4[1:2, 3:end],   # Top Right
        U4[3:end, 3:end] # Bottom Right
    )

    UH = spzeros(ComplexF64,2^L,2^L)
    @fastmath @inbounds for (idx, quadrant) in enumerate(quadrants)
        a = zeros(2,2)
        a[idx] = 1.

        UH .+= quadrant ⊗ speye(2^(L-2)) ⊗ a
    end
    return UH
end

function random_haar_bulk(site::Int, L::Int, quadrants::NTuple{4, SubArray{ComplexF64, 2, Matrix{ComplexF64}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}})
    UH = spzeros(ComplexF64,2^4,2^4)
    @fastmath @inbounds for (idx, quadrant) in enumerate(quadrants)
        a = zeros(2,2)
        a[idx] = 1.

        UH .+= a ⊗ speye(4) ⊗ quadrant
    end
    return speye(2^(site-1)) ⊗ UH ⊗ speye(2^(L-site-3))
end

function random_haar_UB_other_side(L::Int,quadrants::NTuple{4, SubArray{ComplexF64, 2, Matrix{ComplexF64}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}})
    UH = spzeros(ComplexF64,2^(L-2),2^(L-2))
    for (idx, quadrant) in enumerate(quadrants)
        a = zeros(2,2)
        a[idx] = 1.

        UH += quadrant ⊗ speye(2^(L-4)) ⊗ a
    end
    return UH⊗ speye(4)
end

function random_haar_proj_other_side(L::Int,quadrants::NTuple{4, SubArray{ComplexF64, 2, Matrix{ComplexF64}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}})
    UH = spzeros(ComplexF64,2^(L-2),2^(L-2))
    @fastmath @inbounds for (idx, quadrant) in enumerate(quadrants)
        a = zeros(2,2)
        a[idx] = 1.

        UH .+=  quadrant ⊗ speye(2^(L-4)) ⊗ a
    end
    return speye(4) ⊗ UH
end

# function random_haar_proj_splitted(L::Int,quadrants::NTuple{4, SubArray{ComplexF64, 2, Matrix{ComplexF64}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}})
#     UH = spzeros(ComplexF64,2^L,2^L)
#     @fastmath @inbounds for (idx, quadrant) in enumerate(quadrants)
#         a = zeros(2,2)
#         a[idx] = 1.

#         UH .+= speye(2) ⊗ quadrant ⊗ speye(2^(L-4)) ⊗ a ⊗ speye(2)
#     end
#     return UH
# end

function random_haar_proj_splitted(L::Int,quadrants::NTuple{4, SubArray{ComplexF64, 2, Matrix{ComplexF64}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}})
    UH = spzeros(ComplexF64,2^(L-2),2^(L-2))
    for (idx, quadrant) in enumerate(quadrants)
        a = zeros(ComplexF64,2,2)
        a[idx] = 1.

        UH +=  quadrant ⊗ speye(2^(L-4)) ⊗ a 
    end
    return speye(2) ⊗ UH ⊗ speye(2)
end

# function haar_measure(n::Int) :: Matrix{ComplexF64}
#     z = randn(ComplexF64,n,n)/sqrt(2)
#     q,r = qr(z)
#     r./=abs.(r)
#     return q * Diagonal(r)
# end

function haar_measure(n::Int)::Matrix{ComplexF64}
    Z = randn(ComplexF64, n, n)                 
    F = qr!(Z)                                  
    Q = Matrix(F.Q)                              
    d = diag(F.R)                                
    @inbounds for j in 1:n
        s = sign(d[j])
        @views Q[:, j] .*= s
    end
    return Q
end

# random U(1) preserving gate according to 10.1103/PhysRevX.12.041002
function u1_preserving_2qubit_haar()::Matrix{ComplexF64}
    U = spzeros(ComplexF64,4,4)
    U[1,1] = haar_measure(1)[1,1]  # 00 -> 00
    U[4,4] = haar_measure(1)[1,1]  # 11 -> 11
    U[2:3, 2:3] = haar_measure(2)  # (01,10) -> (01,10)
    return U
end