using MonitoredSpinChains
import MonitoredSpinChains: mpiexec

nworker = 5

p = run(`$(mpiexec()) -n $nworker $(Base.julia_cmd()) full_sim.jl`)


# include("full_sim.jl")
# include("fs1.jl")