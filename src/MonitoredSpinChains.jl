module MonitoredSpinChains

    using LinearAlgebra
    using SparseArrays
    using JLD2
    using LuxurySparse: IMatrix


    include("struct.jl")
    include("constants.jl")
    include("observables.jl")
    include("mpiSimulation.jl")
    include("dynamics.jl")
end