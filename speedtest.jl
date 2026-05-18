#!/usr/bin/env julia
# Run from the MonitoredSpinChains.jl root: julia --project=. speedtest.jl

# ══════════════════════════════════════════════════════════════════════════════
#  MonitoredSpinChains speed test  (full 2^L Hilbert space)
#  Times bare time_step! calls — no disk I/O, no observable recording.
# ══════════════════════════════════════════════════════════════════════════════

# ── Simulation parameters ──────────────────────────────────────────────────────
L            = 20           # system size  (state vector: 2^L)
bc           = :pbc         # :obc  or  :pbc
unitaryRate  = 100.0          # gates per step
measurement  = true         # apply measurements
noise        = 0.0

# ── Timing ─────────────────────────────────────────────────────────────────────
n_warmup = 10             # steps to discard (JIT warmup)
n_steps  = 100            # steps to time

# ── Override from combined speedtest.sh (via environment variables) ───────────
L        = parse(Int, get(ENV, "SPEEDTEST_L",        string(L)))
n_steps  = parse(Int, get(ENV, "SPEEDTEST_N_STEPS",  string(n_steps)))
n_warmup = parse(Int, get(ENV, "SPEEDTEST_N_WARMUP", string(n_warmup)))

# ══════════════════════════════════════════════════════════════════════════════

using MonitoredSpinChains
import MonitoredSpinChains: SU2Trajectory, SU2PBCTrajectory, neelState,
                             compute_missing_parameters!, time_step!
using Printf

TrajType = (bc == :pbc) ? SU2PBCTrajectory : SU2Trajectory

circuit = Circuit(
    L,
    n_steps,        # meas_steps  (unused here)
    1,              # average
    Float64(unitaryRate),
    :twoSpinHaar,   # unitarySetup (unused with unitaryRate=0)
    bc,
    rand_spinhalf_im,      # old neelState takes only L::Int
    measurement,
    :Z,          # feedback
    Float64(noise),
    tempdir(),      # result_folder — never written to
    [:OP],          # observables
    false,          # trajectories_averaged
    0,              # thermalizationSteps
    1,              # meas_every
    "su2",
)

traj = TrajType(
    trajectoryID     = 1,
    circuit          = circuit,
    current_timestep = 1,
    thermalized      = true,
)
compute_missing_parameters!(traj)

nstates = 2^L

# ── Warmup (JIT) ──────────────────────────────────────────────────────────────
psi_initial = copy(traj.state)
for _ in 1:n_warmup
    time_step!(traj)
end
traj.state .= psi_initial   # restore Néel state before timed run

# ── Timed run ─────────────────────────────────────────────────────────────────
t_start = time_ns()
for _ in 1:n_steps
    time_step!(traj)
end
elapsed     = (time_ns() - t_start) / 1e9
per_step_μs = elapsed / n_steps * 1e6

# ── Print results ─────────────────────────────────────────────────────────────
println("══════════════════════════════════════════════════════════════")
println(" Julia (MonitoredSpinChains)   full 2^L Hilbert space")
@printf(" L=%-4d  Sztot=%-4d  nstates=%-10d  [ serial ]\n", L, 0, nstates)
println("──────────────────────────────────────────────────────────────")
@printf(" %-12s: %9d\n",    "warmup",     n_warmup)
@printf(" %-12s: %9d\n",    "steps",      n_steps)
@printf(" %-12s: %9.3f s\n",  "total time", elapsed)
@printf(" %-12s: %9.3f μs\n", "per step",   per_step_μs)
println("══════════════════════════════════════════════════════════════")
