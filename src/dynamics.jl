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

function meas!(traj::Trajectory, site::Int)
    Ppsi = traj.projectors[site] * traj.state
    prob = real(dot(traj.state, Ppsi))
    if rand()<prob
        sqrtProb = sqrt(prob)
        traj.state .= Ppsi/sqrtProb

        # false not correction if measurement outcome is singlet
        if (rand() < (1.0+exp(-traj.circuit.noise))/2.) && (site == 1)
            correct!(traj,site)
        end
    else
        traj.state .= (traj.state - Ppsi)/sqrt(1.0-prob)
        # false correction if measurement outcome is triplet
        if rand() < (1.0-exp(-traj.circuit.noise))/2.
            correct!(traj,site)
        end 
    end

    return 
end

# Specialized fast SU2 measurement dispatch (replaces meas_fast! wrapper)
# function meas!(traj::SU2Trajectory, site::Int)
#     _meas_fast_su2!(traj, site)
# end

# function meas!(traj::SU2PBCTrajectory, site::Int)
#     _meas_fast_su2!(traj, site)
# end

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
    _meas_fast_su2_core!(psi, L, i, j)
    return traj
end

# Core, allocation-free kernel acting in-place on psi for a single nearest-neighbour bond (i,j)
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
    return nothing
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

# Fast Fredkin PBC measurement (3-qubit projector) for sites 1..L-2; falls back otherwise
function meas_fast_fredkin!(traj::FredkinPBCTrajectory, site::Int)
    L = traj.circuit.L
    if 1 <= site <= L-2
        psi = traj.state::Vector{ComplexF64}
        p = _meas_fast_fredkin_core!(psi, L, site, site+1, site+2)
        # noise logic analogous to meas!: treat success probability = p
        if rand() < p
            if rand() < (1 + exp(-traj.circuit.noise)) * 0.5
                correct!(traj, site)
            end
        else
            if rand() < (1 - exp(-traj.circuit.noise)) * 0.5
                correct!(traj, site)
            end
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
        dj = d[j]
        phase = dj == 0 ? one(dj) : dj/abs(dj)  
        @views Q[:, j] .*= phase
    end
    return Q
end