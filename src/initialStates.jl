export rand_spinhalf_im,
    rand_spinone_im,
    rand_spinhalf_real,
    rand_spinone_real,
    neelState,
    anomalous_ground_state,
    flat_spin1,
    generalized_dicke,
    normalize_initial_state,
    serialize_initial_state,
    normalize_step_function,
    serialize_step_function,
    StepFunction


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

# Registry mapping short names to initial-state constructors. Placed after
# definitions to avoid forward-reference issues during module initialization.
const INITIAL_STATE_REGISTRY = Dict(
    "rand_spinhalf_im"           => rand_spinhalf_im,
    "rand_spinone_im"            => rand_spinone_im,
    "rand_spinhalf_real"         => rand_spinhalf_real,
    "rand_spinone_real"          => rand_spinone_real,
    "neelState"                  => neelState,
    "neelState1"                 => neelState1,
    "flat_spin1"                 => flat_spin1,
    "fredkin_stationary_state"   => fredkin_stationary_state,
)

# Inverse lookup for serialization
const INV_INITIAL_STATE_REGISTRY = Dict(v => k for (k, v) in INITIAL_STATE_REGISTRY)

"""Normalize an initial-state identifier into a callable function."""
function normalize_initial_state(x)
    if x isa Function
        return x
    elseif x isa Symbol
        haskey(INITIAL_STATE_REGISTRY, String(x)) || error("Unknown initial state identifier: $(x)")
        return INITIAL_STATE_REGISTRY[String(x)]
    elseif x isa AbstractString
        haskey(INITIAL_STATE_REGISTRY, x) || error("Unknown initial state identifier: $(x)")
        return INITIAL_STATE_REGISTRY[x]
    else
        error("Unsupported initial state type: $(typeof(x))")
    end
end

"""Serialize a known initial-state function back to its registry key."""
function serialize_initial_state(f::Function)
    haskey(INV_INITIAL_STATE_REGISTRY, f) || error("Initial state function not in registry: $(f)")
    return INV_INITIAL_STATE_REGISTRY[f]
end

# Wrapper to carry a serializable spec for step-like functions (meas_steps, thermalizationSteps, meas_every)
struct StepFunction{F}
    f::F
    spec::NamedTuple
end

(sf::StepFunction)(L::Int) = sf.f(L)

Base.hash(sf::StepFunction, h::UInt) = hash(sf.spec, h)
Base.show(io::IO, sf::StepFunction) = print(io, "StepFunction(", sf.spec, ")")

"""Normalize a step spec into a callable StepFunction.

Supported specs (all serializable by JLD2):
  - Number: treated as constant → coeff=number, power=0, divisor=1
  - NamedTuple/Dict with keys :power (default 0), :divisor (default 1), :coeff (default 1)
  - Function: allowed in-memory; serialization will fail unless expressed as a spec
"""
function normalize_step_function(x)
    if x isa StepFunction
        return x
    elseif x isa Number
        spec = (; power=0, divisor=1, coeff=x)
        return StepFunction(_ -> x, spec)
    elseif x isa NamedTuple
        spec = canonical_step_spec(x)
        return StepFunction(step_callable(spec), spec)
    elseif x isa Dict
        spec = canonical_step_spec(NamedTuple(Symbol(k) => v for (k,v) in x))
        return StepFunction(step_callable(spec), spec)
    elseif x isa Function
        spec = (; kind=:raw_function, name=string(x))
        return StepFunction(x, spec)
    else
        error("Unsupported step-function type: $(typeof(x))")
    end
end

"""Serialize a step function or spec into a JLD2-safe object."""
function serialize_step_function(x)
    sf = x isa StepFunction ? x : normalize_step_function(x)
    spec = sf.spec

    if spec isa NamedTuple
        if haskey(spec, :kind) && spec.kind == :raw_function
            error("Cannot serialize arbitrary function $(spec.name); provide a (power, divisor, coeff) spec instead")
        end
        return spec
    else
        error("Unsupported step-function spec for serialization: $(spec)")
    end
end

# Normalize and validate a step spec into (power, divisor, coeff)
function canonical_step_spec(spec::NamedTuple)
    p = get(spec, :power, 0)
    d = get(spec, :divisor, 1)
    c = get(spec, :coeff, 1)
    d == 0 && error("divisor must be nonzero")
    return (; power=p, divisor=d, coeff=c)
end

# Build callable for a given spec
function step_callable(spec::NamedTuple)
    p = spec.power
    d = spec.divisor
    c = spec.coeff
    return (L -> c*div(L^p, d))
end