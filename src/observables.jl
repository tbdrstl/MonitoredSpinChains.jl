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
    end
    return observables
end

function total_projector(traj::Trajectory)
    @unpack L,bc = traj.circuit

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
                exx = pauli2_expectation(psi, L, i, j, :X)
                eyy = pauli2_expectation(psi, L, i, j, :Y)
                ezz = pauli2_expectation(psi, L, i, j, :Z)
                p = 0.25 * (1 - exx - eyy - ezz)
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
                exx = pauli2_expectation(psi, L, i, j, :X)
                eyy = pauli2_expectation(psi, L, i, j, :Y)
                ezz = pauli2_expectation(psi, L, i, j, :Z)
                p = 0.25 * (1 - exx - eyy - ezz)
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
            exx = pauli2_expectation(psi, L, i, j, :X)
            eyy = pauli2_expectation(psi, L, i, j, :Y)
            ezz = pauli2_expectation(psi, L, i, j, :Z)
            p = 0.25 * (1 - exx - eyy - ezz)
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
