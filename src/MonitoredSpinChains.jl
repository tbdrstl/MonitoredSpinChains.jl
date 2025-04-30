module MonitoredSpinChains

    using LinearAlgebra
    using SparseArrays
    using JLD2
    using LuxurySparse: IMatrix

    export  create_simulation,
            simulate

    include("struct.jl")
    include("constants.jl")
    include("observables.jl")
    include("mpiSimulation.jl")
    include("dynamics.jl")
    include("initialize.jl")
    include("initialStates.jl")
    
end