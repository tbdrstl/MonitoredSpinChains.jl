using MonitoredSpinChains
using Test
import MonitoredSpinChains: mpiexec




# nworker = 2

# p = @time run(`$(mpiexec()) -n $nworker $(Base.julia_cmd()) --check-bounds=no full_sim.jl`)

# include("normalization.jl")
# include("full_sim.jl")
# include("fs1.jl")
include("ancilla.jl")