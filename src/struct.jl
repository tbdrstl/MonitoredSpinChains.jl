abstract type Trajectory end
abstract type SpinOneTrajectory <: Trajectory end
abstract type SpinHalfTrajectory <: Trajectory end

@kwdef mutable struct Observables
    total_proj        ::Union{Missing,Matrix{Float64}} = missing
    entanglement_entropy::Union{Missing,Vector{Float64}} = missing
    magnetization     ::Union{Missing,Matrix{Float64}} = missing
    magnetizationX    ::Union{Missing,Matrix{Float64}} = missing
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
    result_folder           ::String
    observables             ::Vector{Symbol}                
    trajectories_averaged   ::Bool
    thermalizationSteps     ::Int
    meas_every              ::Int
    model                   ::String
end
struct Simulation
    name            ::String
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
    fnames = fieldnames(typeof(circ))
    to_hash = ""
    for fname_iterator in fnames
        to_hash *= string(getfield(circ, fname_iterator))
    end
    return hash(to_hash)
end

function hash(traj::Trajectory)
    # only hash trajID and circuit of trajectory. All information needed + oher things might change
    to_hash = string(traj.trajectoryID)
    to_hash *= string(hash(traj.circuit))
    return hash(to_hash)
end

function hash(circuit::Circuit, trajID::Int64)
    # only hash trajID and circuit of trajectory. All information needed + oher things might change
    to_hash = string(trajID)
    to_hash *= string(hash(circuit))
    return hash(to_hash)
end
