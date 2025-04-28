function get_observables!(traj::Trajectory)
    if traj.current_timestep % traj.circuit.meas_every != 0
        return traj
    end

    current_meas_step = traj.current_timestep ÷ traj.circuit.meas_every

    for obs in traj.circuit.observables
        obs == :OP && (traj.observables.total_proj[current_meas_step, :] .= total_projector(traj))
    end

    return
end

function get_observables(circuit::Circuit)::Observables
    observables = Observables()
    for obs in circuit.observables
        obs == :OP && (observables.total_proj = zeros(circuit.meas_steps, 2))
    end
    return observables
end

function total_projector(traj::Trajectory)
    @unpack L = traj.circuit

    OP = 0.0
    OPvar = 0.0

    @fastmath @inbounds for site in 1:L
        dotprod = real(dot(traj.state, traj.projectors[site], traj.state))
        OP += dotprod
        OPvar += dotprod^2
        # OP += traj.state' * (traj.projectors[site] * traj.state)
    end
    OP = OP / L
    OPvar = OPvar / L - OP^2

    return OP, OPvar
end