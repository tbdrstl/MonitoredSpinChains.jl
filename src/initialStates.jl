export rand_spinhalf_im,
        rand_spinone_im,
        rand_spinhalf_real,
        rand_spinone_real,
        neelState,
        anomalous_ground_state,
        flat_spin1,
        generalized_dicke, 
        all_plus


function rand_spinhalf_im(L::Int)
    psi = randn(ComplexF64,2^L) 
    return psi ./ norm(psi)
end

function rand_spinone_im(L::Int)
    psi = randn(ComplexF64,3^L)
    return psi ./ norm(psi)
end

function rand_spinhalf_real(L::Int)
    psi = randn(Float64,2^L) 
    return psi ./ norm(psi)
end

function rand_spinone_real(L::Int)
    psi = randn(Float64,3^L)
    return psi ./ norm(psi)
end

function neelState(L::Int64) :: AbstractVector{ComplexF64}
    L == 0 && return [one(ComplexF64)]
    psi = down 
    
    @views for i in 1:L-1
        if iseven(i)
            psi = psi ⊗ down
        else
            psi = psi ⊗ up
        end
    end

    return psi |> Vector{ComplexF64}
end

function neelState1(L::Int) :: AbstractArray{ComplexF64}
    @assert L >= 0 "L must be non-negative"
    if L == 0
        return [one(ComplexF64)]
    end
    psi = down1
    for i in 1:L-1
        if iseven(i)
            psi = psi ⊗ down1
        else
            psi = psi ⊗ up1
        end
    end
    return Vector{ComplexF64}(psi)
end

function all_plus(L::Int; ancilla::Bool=false)
    @assert L >= 0 "L must be non-negative"
    if L == 0
        return [one(ComplexF64)]
    elseif L == 1
        return plus
    end

    psi = plus
    for i in 1:L-2
        psi = psi ⊗ plus
    end
    if ancilla 
        psi = psi ⊗ singlet
    else
        psi = psi ⊗ plus
    end
    return Vector{ComplexF64}(psi)
end

function flat_spin1(L::Int)
    L == 1 && return flat1
    return flat_spin1(L-1) ⊗ flat1 |> Vector{ComplexF64}
end

function anomalous_ground_state(L::Int)
    @assert iseven(L) "L must be even"

    if isfile(joinpath(anomalousstatepath,"anomalousGS$L.bin"))
        println("Loading anomalous state from file")
        return load_anomalous(L)
    end
    
    state = zeros(2^L)
    
    @showprogress for comb in combinations(1:L, div(L,2))
        conf = zeros(Int, L)
        conf[comb] .= 1
        m = abs(minn(2 .* conf .- 1))
        state .+= state_bits_to_vector(conf) * (-1)^m
    end
    state ./= sqrt(binomial(L,div(L,2)))

    # save the state to a file for faster future use
    if !ispath(anomalousstatepath)
        mkpath(anomalousstatepath)
    end
    write(joinpath(anomalousstatepath,"anomalousGS$L.bin"), state)

    return state
end

function minn(vec)
    res = vec[1]
    min_sum = vec[1]
    for i in 2:length(vec)
        res += vec[i]
        if res < min_sum
            min_sum = res
        end
    end
    return min_sum
end

function size_equivalence(L::Int, a::Int, b::Int)
    @assert iseven(L) "L must be even"
    L == a + b && return 1
    if !(iseven(L-a-b) && L-a-b > 0)
        return 0
    end
    return binomial(L,div(L+a+b,2)) - binomial(L,div(L+a+b,2)+1)
end

function state_bit(L,m)
    state_bits = zeros(Int, L)
    k = L-2m
    for i in 1:div(k,2)
        state_bits[2*i-1] = 1
        state_bits[2*i] = 0
    end
    state_bits[k+1:k+m] .= 0
    state_bits[k+m+1:k+2m] .= 1

    return state_bits
end

function state_bits_to_vector(state_bits::Vector{Int})
    res = ifelse(state_bits[1] == 1, up, down)
    for elem in state_bits[2:end]
        if elem == 1
            res = res ⊗ up
        elseif elem == 0
            res = res ⊗ down
        end
    end
    return res
end

function load_dicke(L::Int,k::Int) ::AbstractVector{Float64}
    psi = Vector{Float64}(undef, 2^L)
    open(joinpath(dickestatepath,"dickeStateX_L$(L)_l$(k).bin")) do file
        read!(file, psi)
    end
    return psi
end

function load_anomalous(L::Int) ::AbstractVector{Float64}
    psi = Vector{Float64}(undef, 2^L)
    open(joinpath(anomalousstatepath,"anomalousGS$L.bin")) do file
        read!(file, psi)
    end
    return psi
end

"""
    function generalized_dicke(L::Int, d::Int)

    # Arguments
    L::Int: Length of the chain
    d::Int: Dimension of the spin (e.g., 3 for spin-1)

    Generates all Dicke states of a spin chain of length `L` with 
    `d` spin degree of freedom.
"""
function generalized_dicke(L::Int, d::Int)
    dicke_states = [spzeros(Float64,d^L) for _ in 1:nmultisets(0:d-1, L)]
    for (ind,m) in enumerate(multisets(0:d-1, L))
        mp = multiset_permutations(m,L)
        @showprogress desc="Computing Dicke..." for state_bits in mp
            dicke_states[ind] .+= _state_bits_to_vector_general(state_bits,d)
        end
        dicke_states[ind] ./= sqrt(length(mp))
    end
    return dicke_states
end

function _state_bits_to_vector_general(state_bits,d::Int)
    res = sparsevec(Dict(state_bits[1]+1=>1.), d)
    for i in 2:length(state_bits)
        res = res ⊗ sparsevec(Dict(state_bits[i]+1=>1), d)
    end
    return res
end

"""
    function fredkin_stationary_state(L::Int)

    Generates the stationary state of the Fredkin model for a given system size `L`.
    The state is the antisymmetric superposition of the z=0 Dicke state and the anom-
    alous state. 
"""
function fredkin_stationary_state(L::Int)
    @assert iseven(L) "L must be even"
    a = anomalous_ground_state(L)
    d = generalized_dicke(L, 2)[div(L,2)+1]

    state = (a .- d) ./ sqrt(2)
    return state
end

f_stat_ent(L) = entanglement_entropy_general(fredkin_stationary_state(L), 1:div(L,2))