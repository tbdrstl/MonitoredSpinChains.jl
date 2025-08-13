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
    end
    return observables
end

function total_projector(traj::Trajectory)
    @unpack L = traj.circuit

    OP = 0.0
    OPvar = 0.0

    @fastmath @inbounds for site in 1:L
        dotprod = real(dot(traj.state, traj.projectors[site], traj.state))
        OP += dotprod
        OPvar += dotprod^2
    end
    OP = OP / L
    OPvar = OPvar / L - OP^2
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
