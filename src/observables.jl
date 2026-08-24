function get_observables!(traj::Trajectory)
    if traj.current_timestep % traj.circuit.meas_every != 0
        return traj
    end

    current_meas_step = traj.current_timestep ÷ traj.circuit.meas_every

    for obs in traj.circuit.observables
        obs == :OP && (traj.observables.total_proj[current_meas_step, :] .= total_projector(traj))
        obs == :EE && (traj.observables.entanglement_entropy[current_meas_step] = entanglement_entropy_general(traj,1:div(traj.circuit.L,2)))
        obs == :M && (traj.observables.magnetization[current_meas_step, :] .= magnetization(traj))
        obs == :MX && (traj.observables.magnetizationX[current_meas_step, :] .= magnetizationX(traj))
        obs == :OPH && (traj.observables.total_proj_half[current_meas_step, :] .= projector_stats_r(traj, div(traj.circuit.L,2)))
        obs == :OPQ && (traj.observables.total_proj_quarter[current_meas_step, :] .= projector_stats_r(traj, div(traj.circuit.L,4)))
        obs == :W && (traj.observables.witness[current_meas_step] = total_witness(traj))

        if obs == :EEfin && traj.current_timestep == traj.circuit.meas_steps*traj.circuit.meas_every
            # Calculate the entanglement entropy for the final state
            for lmax in 1:div(traj.circuit.L,2)
                A = 1:lmax
                ent = entanglement_entropy_general(traj, A)
                traj.observables.entanglement_entropy[lmax] = ent
            end
        end
    end

    return
end

function get_observables(circuit::Circuit)::Observables
    observables = Observables()
    for obs in circuit.observables
        obs == :OP && (observables.total_proj = zeros(circuit.meas_steps, 3))
        obs == :EE && (observables.entanglement_entropy = zeros(circuit.meas_steps))
        obs == :M && (observables.magnetization = zeros(circuit.meas_steps, 2))
        obs == :MX && (observables.magnetizationX = zeros(circuit.meas_steps, 2))
        obs == :EEfin && (observables.entanglement_entropy = zeros(div(circuit.L,2)))
        obs == :OPH && (observables.total_proj_half = zeros(circuit.meas_steps,3))
        obs == :OPQ && (observables.total_proj_quarter = zeros(circuit.meas_steps,3))
        obs == :W && (observables.witness = zeros(circuit.meas_steps))
    end
    return observables
end

"""
    total_witness(traj) -> Float64

The preparation witness `W = Σ_{i<j} ⟨P̂_ij⟩` of Eq. (34), summed over all
`L(L-1)/2` unordered pairs.

Evaluated through the collective-spin identity of Eq. (56),

    Ŵ = ½ [ J_max(J_max + 1) − Ĵ² ],    J_max = L/2,

which replaces the O(L²) sum of pair projectors by three global spin operators.
Writing `Ĵ² = (Ĵᶻ)² + ½(Ĵ⁺Ĵ⁻ + Ĵ⁻Ĵ⁺)` and using `(Ĵ^±)† = Ĵ^∓`,

    ⟨Ĵ²⟩ = ⟨(Ĵᶻ)²⟩ + ½ ( ‖Ĵ⁻ψ‖² + ‖Ĵ⁺ψ‖² ),

so this costs O(L·2^L) with one scratch vector instead of O(L²·2^L).

`W` vanishes on the permutation-symmetric (Dicke) manifold and grows with
leakage out of it; the certified infidelity is `ε_cert = 2W/L`, Eq. (103).

Note `Ŵ` is a *linear* observable, so averaging `⟨ψ|Ŵ|ψ⟩` over trajectories
gives `Tr[ρ̂Ŵ]` exactly — no unravelling bias.
"""
function total_witness(traj::SpinHalfTrajectory)
    L = traj.circuit.L
    psi = traj.state
    length(psi) == 1 << L || error(
        "total_witness: state has length $(length(psi)), expected 2^$L = $(1 << L)")
    # Function barrier: `state` is declared as an abstract AbstractVector union on
    # the trajectory types, so calling the kernel through a concretely typed
    # argument is what keeps the inner loops from dispatching dynamically on every
    # element (same reason _meas_fast_su2! asserts Vector{ComplexF64}).
    return _total_witness(psi, L)
end

function _total_witness(psi::AbstractVector{T}, L::Int) where {T<:Number}
    N = length(psi)

    # bit = 1 is |↓⟩ (the convention of controlPsiZ!), so a basis state with
    # `count_ones` down spins has magnetisation m = (L − 2·count_ones)/2.
    jz2 = 0.0
    @inbounds for s in 0:N-1
        m = (L - 2 * count_ones(s)) / 2
        jz2 += abs2(psi[s+1]) * m * m
    end

    # ‖Ĵ⁺ψ‖² and ‖Ĵ⁻ψ‖², with Ĵ⁺ = Σ_ℓ |↑⟩⟨↓|_ℓ clearing one set bit and Ĵ⁻
    # setting one clear bit.
    #
    # The flipped bit is the OUTER loop and each pass walks contiguous
    # half-blocks, so both the read and the two writes stream sequentially. A
    # state-by-state scatter instead touches buf[s ⊻ (1<<b)] at stride 2^b, which
    # misses cache on every high bit and is several times slower once 2^L leaves
    # L3. Both buffers are filled in the same pass so psi is read once per bit.
    up = zeros(T, N)
    dn = zeros(T, N)
    @inbounds for b in 0:L-1
        mask = 1 << b
        step = mask << 1
        for base in 0:step:(N-1)
            @simd for off in 0:mask-1
                lo = base + off + 1        # bit b = 0, i.e. |↑⟩ at site b+1
                hi = lo + mask             # bit b = 1, i.e. |↓⟩
                up[lo] += psi[hi]
                dn[hi] += psi[lo]
            end
        end
    end

    J2 = jz2 + 0.5 * (sum(abs2, up) + sum(abs2, dn))
    Jmax = L / 2
    return 0.5 * (Jmax * (Jmax + 1) - J2)
end

function total_witness(traj::SpinOneTrajectory)
    error("The :W witness is defined for spin-1/2 models only; Eq. (56) relates it to " *
          "the collective spin-1/2 algebra, which has no counterpart for model " *
          "\"$(traj.circuit.model)\".")
end

"""
    singlet_expectation(psi, L, i, j) -> Float64

`⟨ψ|P̂_ij|ψ⟩` for the two-site singlet projector `P̂_ij = ¼(1 − σ⃗_i·σ⃗_j)`,
computed directly from the amplitudes and without materialising any operator.

With `|s⟩ = (|↑↓⟩ − |↓↑⟩)/√2` the projector weight is

    ⟨P̂_ij⟩ = ½ Σ_rest |a_{↑↓} − a_{↓↑}|²,

summed over the 2^(L−2) configurations of the other sites. This is the same
quantity, and the same loop, as the first pass of `_meas_fast_su2_core!`.
O(2^L) per pair with two sequential read streams, against a sparse 2^L×2^L
matvec (or, worse, building the operator first).
"""
function singlet_expectation(psi::AbstractVector, L::Int, i::Int, j::Int)
    i, j = minmax(i, j)
    (1 <= i < j <= L) || error("singlet_expectation: need 1 ≤ i < j ≤ $L, got ($i, $j)")
    length(psi) == 1 << L || error(
        "singlet_expectation: state has length $(length(psi)), expected 2^$L")
    # Function barrier — `state` is an abstract field type on the trajectories.
    return _singlet_expectation(psi, L, i, j)
end

function _singlet_expectation(psi::AbstractVector{T}, L::Int, i::Int, j::Int) where {T<:Number}
    N = length(psi)
    mask_i = UInt64(1) << (L - i)      # site k occupies bit L−k (big-endian)
    mask_j = UInt64(1) << (L - j)
    both = mask_i | mask_j

    p = 0.0
    @inbounds for s in UInt64(0):UInt64(N - 1)
        # visit each 4-state block once, from its (bit_i, bit_j) = (0,1) member
        if (s & both) == mask_j
            d = psi[Int(s) + 1] - psi[Int(s ⊻ both) + 1]
            p += 0.5 * abs2(d)
        end
    end
    return p
end

"""
    nn_bonds(L, bc) -> Vector{Tuple{Int,Int}}

Nearest-neighbour bonds in the order `get_proj_su2` stores them:
`(1,2), …, (L−1,L)` and, under `:pbc`, the wraparound `(1,L)` last. Keeping this
order means bond index ↔ measurement site agrees with `meas!`.
"""
function nn_bonds(L::Int, bc::Symbol)
    bonds = [(site, site + 1) for site in 1:L-1]
    bc == :pbc && push!(bonds, (1, L))
    return bonds
end

# SU(2): the measured projector is the two-site singlet, so ⟨P̂⟩ is available in
# closed form from the amplitudes and no stored operator is needed. Fredkin,
# Motzkin and AKLT keep the generic sparse method below — their projectors are
# three-site / spin-1 objects, not singlets.
function total_projector(traj::Union{SU2Trajectory,SU2PBCTrajectory})
    @unpack L, bc = traj.circuit
    psi = traj.state

    OP = 0.0
    OPvar = 0.0
    @inbounds for (i, j) in nn_bonds(L, bc)
        dotprod = singlet_expectation(psi, L, i, j)
        OP += dotprod
        OPvar += dotprod^2
    end

    L_bound = bc == :pbc ? L : L - 1

    OP = OP / L_bound
    OPvar = OPvar / L_bound - OP^2
    OP2 = OP^2

    return OP, OPvar, OP2
end

function total_projector(traj::Trajectory)
    @unpack L,bc = traj.circuit

    ismissing(traj.projectors) && error(
        "total_projector: this trajectory type still needs stored projectors, but " *
        "traj.projectors is missing. See needs_projectors(circuit) in initialize.jl.")

    OP = 0.0
    OPvar = 0.0

    @fastmath @inbounds for site in eachindex(traj.projectors)
        dotprod = real(dot(traj.state, traj.projectors[site], traj.state))
        OP += dotprod
        OPvar += dotprod^2
    end

    L_bound = bc == :pbc ? L : L - 1

    OP = OP / L_bound
    OPvar = OPvar / L_bound - OP^2
    OP2 = OP^2

    return OP, OPvar, OP2
end

function entanglement_entropy_general(traj::SpinHalfTrajectory, A::AbstractVector{Int})
    return entanglement_entropy_general(traj.state, A)
end

function entanglement_entropy_general(traj::SpinOneTrajectory, A::AbstractVector{Int})
    return entanglement_entropy_general_spinOne(traj.state, A)
end

function entanglement_entropy_general(psi::AbstractVector{T}, A::AbstractVector{Int}) where {T <: Union{Float64, ComplexF64}}
    # Calculate the number of qubits in the system
    n = Int(log2(length(psi)))
    if length(A) < 0 || length(A) > n
        error("Partition A must have at least one qubit and cannot include all qubits.")
    end
    if any(x -> x < 1 || x > n, A)
        error("Partition indices must be between 1 and the total number of qubits.")
    end
    
    # Determine the complement of A (set B)
    B = setdiff(1:n, A)
    
    # Create a permutation to reorder the qubits such that A comes first, then B
    perm = vcat(A, B)
    
    # Reshape psi into an n-dimensional tensor with each dimension of size 2
    reshaped_psi = reshape(psi, ntuple(_ -> 2, n))

    # Permute the dimensions according to the calculated permutation
    permuted_psi = permutedims(reshaped_psi, perm)
    
    # Reshape the permuted state back into a matrix of size (2^length(A)) x (2^length(B))
    reshaped_permuted_psi = reshape(permuted_psi, 2^length(A), 2^length(B))
    
    # Perform Singular Value Decomposition (SVD)
    s = svdvals!(reshaped_permuted_psi)
    
    # Compute the entanglement entropy
    probabilities = s.^2
    non_zero_probs = probabilities[probabilities .> 0]
    entropy = -sum(non_zero_probs .* log.(non_zero_probs))
    
    return entropy
end

function entanglement_entropy_general_spinOne(psi::AbstractVector{T}, A::AbstractVector{Int}) where {T<:Union{Float64, ComplexF64}}
    # Calculate the number of qubits in the system
    n = Int(round(log(3,length(psi))))
    if length(A) < 0 || length(A) > n
        error("Partition A must have at least one qubit and cannot include all qubits.")
    end
    if any(x -> x < 1 || x > n, A)
        error("Partition indices must be between 1 and the total number of qubits.")
    end
    
    # Determine the complement of A (set B)
    B = setdiff(1:n, A)
    
    # Create a permutation to reorder the qubits such that A comes first, then B
    perm = vcat(A, B)
    
    # Reshape psi into an n-dimensional tensor with each dimension of size 3
    reshaped_psi = reshape(psi, ntuple(_ -> 3, n))


    
    # Permute the dimensions according to the calculated permutation
    permuted_psi = permutedims(reshaped_psi, perm)
    
    # Reshape the permuted state back into a matrix of size (3^length(A)) x (3^length(B))
    reshaped_permuted_psi = reshape(permuted_psi, 3^length(A), 3^length(B))
    
    # Perform Singular Value Decomposition (SVD)
    s = svdvals!(reshaped_permuted_psi)
    
    # Compute the entanglement entropy
    probabilities = s.^2
    non_zero_probs = probabilities[probabilities .> 0]
    entropy = -sum(non_zero_probs .* log.(non_zero_probs))
    
    return entropy
end

function magnetization(traj::SpinHalfTrajectory)
    @unpack L = traj.circuit

    M = 0.0
    varM = 0.0


    @fastmath @inbounds for site in 1:L
        m = 0.5*dot(traj.state, speye(2^(site - 1)) ⊗ Z ⊗ speye(2^(L - site)), traj.state)
        
        M += m
        varM += m^2
    end
    M = real(M) / L
    varM = real(varM) / L - M^2

    return M, varM
end

function magnetization(traj::SpinOneTrajectory)
    @unpack L = traj.circuit

    M = 0.0
    varM = 0.0


    @fastmath @inbounds for site in 1:L
        m = 0.5*dot(traj.state, speye(3^(site - 1)) ⊗ Z1 ⊗ speye(3^(L - site)), traj.state)
        
        M += m
        varM += m^2
    end
    M = real(M) / L
    varM = real(varM) / L - M^2

    return M, varM
end

function magnetizationX(traj::SpinOneTrajectory)
    @unpack L = traj.circuit

    M = 0.0
    varM = 0.0

    @fastmath @inbounds for site in 1:L
        m = 0.5*dot(traj.state, speye(3^(site - 1)) ⊗ X1 ⊗ speye(3^(L - site)), traj.state)

        M += m
        varM += m^2
    end
    M = real(M) / L
    varM = real(varM) / L - M^2

    return M, varM
end

function magnetizationX(traj::SpinHalfTrajectory)
    @unpack L = traj.circuit

    M = 0.0
    varM = 0.0

    @fastmath @inbounds for site in 1:L
        m = 0.5*dot(traj.state, speye(2^(site - 1)) ⊗ X ⊗ speye(2^(L - site)), traj.state)

        M += m
        varM += m^2
    end
    M = real(M) / L
    varM = real(varM) / L - M^2

    return M, varM
end

# according to arXiv:2306.13146v2
function exact_dicke_entanglement(L::Int,mag::Int,subsystemSize::Int)
	ent = 0.0
	L_over_mag = binomial(L, mag)
	
	for i in 0:min(subsystemSize, mag)
		binom_i = binomial(subsystemSize, i) * binomial(L-subsystemSize, mag-i)
		ent -= binom_i * log(binom_i/L_over_mag)
	end
	return ent/L_over_mag
end

# according to arXiv:1904.05205
function exact_dyck_entanglement(L::Int, la::Int)
    p(n) = 1 - mod(n,2) # selects even n

    # number of valid Dyck paths between heights h1 and h2 with n steps
    function D(n,h1,h2)
        return (binomial(n, div(n+abs(h1-h2),2)) - binomial(n, div(n+h1+h2,2) + 1)) * p(n+h1+h2)
    end

    ent = 0.0
    for h in 0:min(L-la, la)
        Dquot = D(la,0,h)*D(L-la,h,0) / D(L, 0,0)
        if Dquot == 0
            continue
        end
        ent -= Dquot * log(Dquot)
    end

    return ent
end

function anomalous_ent(L::Int)
    psi = load_anomalous(L)
    ent = entanglement_entropy_general(psi, 1:div(L,2))
    return ent
end

function anomalous_ent(A::AbstractVector{Int})
    ents = zeros(length(A))
    for (ind,l) in enumerate(A)
        psi = load_anomalous(l)
        ents[ind] = entanglement_entropy_general(psi, 1:div(l,2))
    end   
    return ents
end

# compute averaged projector statistics for pairs separated by distance r (handles PBC & OBC)
function projector_stats_r(traj::SpinHalfTrajectory, r::Int)
    L = traj.circuit.L
    bc = traj.circuit.bc

    if bc == :pbc
        @assert 0 < r <= div(L,2) "For PBC, r must be in (0, L/2]"
        if r == div(L,2) && isodd(L)
            error("r = L/2 only meaningful for even L (PBC)")
        end
    else # :obc
        @assert 0 < r <= L-1 "For OBC, r must be in (0, L-1]"
    end

    psi = traj.state

    sumP = 0.0
    sumP2 = 0.0
    npairs = 0

    if bc == :pbc
        if r == div(L,2) && iseven(L)
            # exactly L/2 unique pairs (i, i+L/2)
            @inbounds for i in 1:div(L,2)
                j = i + r
                p = singlet_expectation(psi, L, i, j)
                sumP += p
                sumP2 += p^2
            end
            npairs = div(L,2)
        else
            # L distinct unordered pairs (i, i+r mod L) since r != L/2
            @inbounds for i in 1:L
                j = i + r
                if j > L
                    j -= L
                end
                # no duplicates for r != L/2
                p = singlet_expectation(psi, L, i, j)
                sumP += p
                sumP2 += p^2
            end
            npairs = L
        end
    else
        # OBC: only pairs fully inside chain: (i, i+r) with i+r <= L
        last_i = L - r
        @inbounds for i in 1:last_i
            j = i + r
            p = singlet_expectation(psi, L, i, j)
            sumP += p
            sumP2 += p^2
        end
        npairs = last_i
    end

    meanP = sumP / npairs
    varP = sumP2 / npairs - meanP^2
    return (meanP, varP, meanP^2)
end

projector_stats_r(::SpinOneTrajectory, r::Int) = error("OPH/OPQ only defined for spin-1/2 trajectories")

function pauli2_expectation(psi::AbstractVector{T}, L::Int, i::Int, j::Int, axis::Symbol) where {T}
    op = pauli_two_site_operator(L, i, j, axis)
    return real(dot(psi, op*psi))
end

function pauli_two_site_operator(L::Int, i::Int, j::Int, axis::Symbol)
    A = axis === :X ? X : axis === :Y ? Y : axis === :Z ? Z : error("Unknown axis $axis")
    if j < i
        i,j = j,i
    end
    left  = i > 1      ? speye(2^(i-1))      : speye(1)
    mid   = j-i-1 > 0  ? speye(2^(j-i-1))    : speye(1)
    right = j < L      ? speye(2^(L-j))      : speye(1)
    return left ⊗ A ⊗ mid ⊗ A ⊗ right
end
