abstract type Trajectory end
abstract type SpinOneTrajectory <: Trajectory end
abstract type SpinHalfTrajectory <: Trajectory end

@kwdef mutable struct Observables
    total_proj        ::Union{Missing,Matrix{Float64}} = missing
    entanglement_entropy::Union{Missing,Vector{Float64}} = missing
    magnetization     ::Union{Missing,Matrix{Float64}} = missing
    magnetizationX    ::Union{Missing,Matrix{Float64}} = missing
    total_proj_half   ::Union{Missing,Matrix{Float64}} = missing   # (mean,var,mean^2) for r = L/2
    total_proj_quarter::Union{Missing,Matrix{Float64}} = missing   # (mean,var,mean^2) for r = L/4
    witness           ::Union{Missing,Vector{Float64}} = missing   # :W — W = Σ_{i<j} ⟨P̂_ij⟩
end

struct Circuit 
    L                       ::Int
    meas_steps              ::Int
    average                 ::Int
    unitaryRate             ::Float64
    unitarySetup            ::Symbol
    bc                      ::Symbol
    initialState            ::Function
    measurement             ::Bool
    feedback                ::Symbol
    noise                   ::NTuple{3,Float64}
    result_folder           ::String
    observables             ::Vector{Symbol}
    trajectories_averaged   ::Bool
    thermalizationSteps     ::Int
    meas_every              ::Int
    model                   ::String
    kick                    ::Symbol
end

# Backward-compatible constructor: call sites predating the `kick` field build a
# Circuit with the 16-argument signature. This forwards to the full 17-argument
# constructor with kick = :none, so those circuits keep their hash (`:kick` is
# registered in DEFAULT_FIELD_VALUES with exactly that default). `noise` is
# accepted as a bare rate here too, so old `Circuit(..., 0.0, ...)` calls work.
function Circuit(L, meas_steps, average, unitaryRate, unitarySetup, bc,
                 initialState, measurement, feedback, noise, result_folder,
                 observables, trajectories_averaged, thermalizationSteps,
                 meas_every, model)
    return Circuit(L, meas_steps, average, unitaryRate, unitarySetup, bc,
                   initialState, measurement, feedback, noise,
                   result_folder, observables, trajectories_averaged,
                   thermalizationSteps, meas_every, model, :none)
end

# Accept a bare rate for `noise` in the full 17-argument form too, so hand-built
# circuits can write `..., 1e-3, ...` and get isotropic noise. More specific than
# the generated all-Any constructor at that one position, so there is no
# ambiguity; an NTuple argument falls through to the generated one unchanged.
function Circuit(L, meas_steps, average, unitaryRate, unitarySetup, bc,
                 initialState, measurement, feedback, noise::Real, result_folder,
                 observables, trajectories_averaged, thermalizationSteps,
                 meas_every, model, kick, kick_step=0)
    return Circuit(L, meas_steps, average, unitaryRate, unitarySetup, bc,
                   initialState, measurement, feedback, normalize_noise(noise),
                   result_folder, observables, trajectories_averaged,
                   thermalizationSteps, meas_every, model, kick, kick_step)
end

# 17-argument form with an explicit noise triple but no kick_step.
function Circuit(L, meas_steps, average, unitaryRate, unitarySetup, bc,
                 initialState, measurement, feedback, noise::NTuple{3,Real},
                 result_folder, observables, trajectories_averaged,
                 thermalizationSteps, meas_every, model, kick)
    return Circuit(L, meas_steps, average, unitaryRate, unitarySetup, bc,
                   initialState, measurement, feedback, normalize_noise(noise),
                   result_folder, observables, trajectories_averaged,
                   thermalizationSteps, meas_every, model, kick, 0)
end

"""
    normalize_noise(x) -> NTuple{3,Float64}

Bring a user-supplied noise specification into the canonical per-site Pauli-rate
triple `(κ_x, κ_y, κ_z)` of Eq. (97).

A bare number `κ` means *isotropic* random-Pauli noise at total rate `κ` per
site, i.e. `(κ/3, κ/3, κ/3)` — each of the three Pauli channels fires at `κ/3`,
matching `𝓛_imp ρ = (κ/3) Σ_{i,a} (σ_i^a ρ σ_i^a − ρ)`. A 3-tuple is taken
verbatim, which is how anisotropic models are expressed (e.g. `(0, 0, κ_z)` is
the fixed-axis dephasing channel of Sec. VI E).
"""
normalize_noise(x::Real) = (Float64(x) / 3, Float64(x) / 3, Float64(x) / 3)
normalize_noise(x::NTuple{3,Real}) = (Float64(x[1]), Float64(x[2]), Float64(x[3]))
function normalize_noise(x::AbstractVector)
    length(x) == 3 || throw(ArgumentError(
        "noise given as a vector must have exactly 3 entries (κ_x, κ_y, κ_z), got $(length(x))"))
    return normalize_noise((x[1], x[2], x[3]))
end
struct Simulation
    params          ::Vector{Circuit}
    params_dict     ::Dict
end

@kwdef mutable struct SU2Trajectory <: SpinHalfTrajectory
    trajectoryID    ::Int64
    circuit         ::Circuit
    current_timestep::Int64
    thermalized     ::Bool
    observables     ::Union{Missing,Observables} = missing
    state           ::Union{Missing,AbstractVector{<:Union{Float64, ComplexF64}}} = missing
    projectors      ::Union{Missing,Vector{SparseMatrixCSC{ComplexF64, Int}}} = missing
    # zFeedbackIndices ::Union{Missing,Vector{Vector{Int32}}} = missing
end

@kwdef mutable struct SU2PBCTrajectory <: SpinHalfTrajectory
    trajectoryID    ::Int64
    circuit         ::Circuit
    current_timestep::Int64
    thermalized     ::Bool
    observables     ::Union{Missing,Observables} = missing
    state           ::Union{Missing,AbstractVector{<:Union{Float64, ComplexF64}}} = missing
    projectors      ::Union{Missing,Vector{SparseMatrixCSC{ComplexF64, Int}}} = missing
    # zFeedbackIndices ::Union{Missing,Vector{Vector{Int32}}} = missing
end

@kwdef mutable struct FredkinTrajectory <: SpinHalfTrajectory
    trajectoryID    ::Int64
    circuit         ::Circuit
    current_timestep::Int64
    thermalized     ::Bool
    observables     ::Union{Missing,Observables} = missing
    state           ::Union{Missing,AbstractVector{<:Union{Float64, ComplexF64}}} = missing
    projectors      ::Union{Missing,Vector{SparseMatrixCSC{ComplexF64, Int}}} = missing
    # zFeedbackIndices ::Union{Missing,Vector{Vector{Int32}}} = missing
end

@kwdef mutable struct FredkinPBCTrajectory <: SpinHalfTrajectory
    trajectoryID    ::Int64
    circuit         ::Circuit
    current_timestep::Int64
    thermalized     ::Bool
    observables     ::Union{Missing,Observables} = missing
    state           ::Union{Missing,AbstractVector{<:Union{Float64, ComplexF64}}} = missing
    projectors      ::Union{Missing,Vector{SparseMatrixCSC{ComplexF64, Int}}} = missing
    # zFeedbackIndices ::Union{Missing,Vector{Vector{Int32}}} = missing
end

@kwdef mutable struct AKLTPBCTrajectory <: SpinOneTrajectory
    trajectoryID    ::Int64
    circuit         ::Circuit
    current_timestep::Int64
    thermalized     ::Bool
    state           ::Union{Missing,AbstractVector{<:Union{Float64, ComplexF64}}} = missing
    observables     ::Union{Missing,Observables} = missing
    projectors      ::Union{Missing,Vector{SparseMatrixCSC{ComplexF64, Int}}} = missing
end

@kwdef mutable struct MotzkinTrajectory <: SpinOneTrajectory
    trajectoryID    ::Int64
    circuit         ::Circuit
    current_timestep::Int64
    thermalized     ::Bool
    state           ::Union{Missing,AbstractVector{<:Union{Float64, ComplexF64}}} = missing
    observables     ::Union{Missing,Observables} = missing
    projectors      ::Union{Missing,Vector{SparseMatrixCSC{ComplexF64, Int}}} = missing
end

@kwdef mutable struct MotzkinPBCTrajectory <: SpinOneTrajectory
    trajectoryID    ::Int64
    circuit         ::Circuit
    current_timestep::Int64
    thermalized     ::Bool
    state           ::Union{Missing,AbstractVector{<:Union{Float64, ComplexF64}}} = missing
    observables     ::Union{Missing,Observables} = missing
    projectors      ::Union{Missing,Vector{SparseMatrixCSC{ComplexF64, Int}}} = missing
end

function add!(a::Observables,b::Observables) ::Observables
    for field in fieldnames(Observables)
        setfield!(a, field, getfield(a, field) + getfield(b, field))
    end 
    return a
end

function divide!(a::Observables,b::Number) ::Observables
    for field in fieldnames(Observables)
        setfield!(a, field, getfield(a, field) / b)
    end 
    return a
end

function square!(a::Observables) ::Observables
    for field in fieldnames(Observables)
        setfield!(a, field, getfield(a, field).^2)
    end 
    return a
end

function sqrt!(a::Observables) ::Observables
    for field in fieldnames(Observables)
        newfield = getfield(a, field)
        if ismissing(newfield)
            continue
        end
        # assume negative sqrt values can only appear for very small values 
        for i in eachindex(newfield)
            if newfield[i] > 0
                newfield[i] = sqrt(newfield[i])
            else 
                newfield[i] = 0.
            end
        end
        setfield!(a, field, newfield)
    end 
    return a
end

function subtract!(a::Observables,b::Observables) ::Observables
    for field in fieldnames(Observables)
        setfield!(a, field, getfield(a, field) - getfield(b, field))
    end 
    return a
end

function hash(circ::Circuit)
    # Build a canonical signature, skipping volatile fields and eliding defaults
    sig = canonical_circuit_signature(circ)
    return hash(sig)
end

function hash(traj::Trajectory)
    # only hash trajID and circuit of trajectory. All information needed + oher things might change
    return hash((traj.trajectoryID, hash(traj.circuit)))
end

function hash(circuit::Circuit, trajID::Int64)
    # only hash trajID and circuit of trajectory. All information needed + oher things might change
    return hash((trajID, hash(circuit)))
end

# Helpers for stable circuit hashing
normalize_for_hash(x::Function) = string(x)
normalize_for_hash(x::Symbol) = String(x)
normalize_for_hash(x::AbstractString) = String(x)
normalize_for_hash(x::AbstractVector) = map(normalize_for_hash, x)
normalize_for_hash(x::Dict) = [(k, normalize_for_hash(x[k])) for k in sort(collect(keys(x)))]
# `noise` used to be a bare Float64, and every pre-existing circuit signature
# therefore carries the entry (:noise, 0.0). Collapsing the all-zero triple back
# to that scalar keeps every noiseless circuit's hash byte-identical to what it
# was before `noise` became a per-Pauli rate triple, so previously computed data
# in result folders stays addressable.
normalize_for_hash(x::NTuple{3,Float64}) = all(iszero, x) ? 0.0 : x
normalize_for_hash(x) = x

function canonical_circuit_signature(circ::Circuit)
    sig = Vector{Any}()
    for fname in fieldnames(Circuit)
        if fname in MonitoredSpinChains.VOLATILE_FIELDS
            continue
        end
        if haskey(MonitoredSpinChains.DEFAULT_FIELD_VALUES, fname)
            default_val = MonitoredSpinChains.DEFAULT_FIELD_VALUES[fname]
            if getfield(circ, fname) == default_val
                continue
            end
        end
        push!(sig, (fname, normalize_for_hash(getfield(circ, fname))))
    end
    return sig
end
