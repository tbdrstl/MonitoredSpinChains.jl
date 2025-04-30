using MonitoredSpinChains
import MonitoredSpinChains: mpiexec

nworker = 5

p = run(`$(mpiexec()) -n $nworker $(Base.julia_cmd()) fs1.jl`)


# include("full_sim.jl")
# include("fs1.jl")