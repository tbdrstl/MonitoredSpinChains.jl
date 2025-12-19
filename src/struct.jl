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
end

struct Circuit 
    L                       ::Int
    meas_steps              ::Int
    average                 ::Int
    # unitaryRate             ::A where A<:Real
    # unitarySetup            ::Symbol
    bc                      ::Symbol
    initialState            ::Function
    measurement             ::Bool
    feedback                ::Symbol  
    noise                   ::Real
    result_folder           ::String
    observables             ::Vector{Symbol}                
    trajectories_averaged   ::Bool
    thermalizationSteps     ::Int
    meas_every              ::Int
    model                   ::String
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
    projectors      ::Union{Missing,Vector{SparseMatrixCSC{Float64, Int}}} = missing
    # zFeedbackIndices ::Union{Missing,Vector{Vector{Int32}}} = missing
end

@kwdef mutable struct SU2PBCTrajectory <: SpinHalfTrajectory
    trajectoryID    ::Int64
    circuit         ::Circuit
    current_timestep::Int64
    thermalized     ::Bool
    observables     ::Union{Missing,Observables} = missing
    state           ::Union{Missing,AbstractVector{<:Union{Float64, ComplexF64}}} = missing
    projectors      ::Union{Missing,Vector{SparseMatrixCSC{Float64, Int}}} = missing
    # zFeedbackIndices ::Union{Missing,Vector{Vector{Int32}}} = missing
end

@kwdef mutable struct FredkinTrajectory <: SpinHalfTrajectory
    trajectoryID    ::Int64
    circuit         ::Circuit
    current_timestep::Int64
    thermalized     ::Bool
    observables     ::Union{Missing,Observables} = missing
    state           ::Union{Missing,AbstractVector{<:Union{Float64, ComplexF64}}} = missing
    projectors      ::Union{Missing,Vector{SparseMatrixCSC{Float64, Int}}} = missing
    # zFeedbackIndices ::Union{Missing,Vector{Vector{Int32}}} = missing
end

@kwdef mutable struct FredkinPBCTrajectory <: SpinHalfTrajectory
    trajectoryID    ::Int64
    circuit         ::Circuit
    current_timestep::Int64
    thermalized     ::Bool
    observables     ::Union{Missing,Observables} = missing
    state           ::Union{Missing,AbstractVector{<:Union{Float64, ComplexF64}}} = missing
    projectors      ::Union{Missing,Vector{SparseMatrixCSC{Float64, Int}}} = missing
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
