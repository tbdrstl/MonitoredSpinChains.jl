module MonitoredSpinChains

    using LinearAlgebra
    using SparseArrays
    using JLD2
    using LuxurySparse: IMatrix
    using UnPack
    using MPI
    using MKL
    import Combinatorics: combinations, multiset_permutations
    import Combinat: multisets, nmultisets
    import ProgressMeter: @showprogress
    import Base: hash
    import Pkg: dependencies
    import Random: randperm

    export  Simulation,
            Circuit,
            SpinHalfTrajectory,
            SpinOneTrajectory,
            FredkinTrajectory,
            FredkinPBCTrajectory,
            MotzkinTrajectory,
            MotzkinPBCTrajectory,
            create_simulation,
            simulate,
            get_trajectories_from_simulation,
            get_trajectories_from_circuit,
            get_projectors,
            ⊗

    include("struct.jl")
    include("constants.jl")
    include("observables.jl")
    include("mpiSimulation.jl")
    include("dynamics.jl")
    include("initialize.jl")
    include("initialStates.jl")
    include("fileIO.jl")
    include("analysis.jl")

end