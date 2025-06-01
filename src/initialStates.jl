export rand_spinhalf_im,
        rand_spinone_im,
        rand_spinhalf_real,
        rand_spinone_real,
        neelState,
        anomalous_ground_state,
        flat_spin1


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

function neelState(L::Int64) :: AbstractVector{Float64}
    L == 0 && return [one(ComplexF64)]
    psi = down 
    
    @views for i in 1:L-1
        if iseven(i)
            psi = psi ⊗ down
        else
            psi = psi ⊗ up
        end
    end

    return psi |> Vector{Float64}
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

function load_anomalous(L::Int) ::AbstractVector{Float64}
    psi = Vector{Float64}(undef, 2^L)
    open(joinpath(anomalousstatepath,"anomalousGS$L.bin")) do file
        read!(file, psi)
    end
    return psi
end

