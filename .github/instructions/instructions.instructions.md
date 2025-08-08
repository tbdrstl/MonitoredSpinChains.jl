---
applyTo: '**'
---
# MonitoredSpinChains Codebase Structure and Guidelines

## Overview
This Julia package simulates quantum spin chains with monitored dynamics, focusing on spin-1 and spin-1/2 models. It is designed for extensibility, performance, and clarity, supporting both open and periodic boundary conditions, and parallel execution via MPI.

## Directory and File Structure
- `src/` — Main source code, modularized by functionality:
  - `MonitoredSpinChains.jl`: Main module, imports dependencies, exports API, and includes all submodules.
  - `struct.jl`: Defines core data structures (e.g., `Simulation`, `Circuit`, `Trajectory` types).
  - `constants.jl`: Physical and mathematical constants, operators (Pauli matrices, projectors, etc.).
  - `initialStates.jl`: Functions to generate initial quantum states (random, Néel, Dicke, etc.).
  - `initialize.jl`: Projector construction, simulation/circuit setup, and trajectory management.
  - `dynamics.jl`: Time evolution, measurement, feedback, and correction routines for trajectories.
  - `observables.jl`: Observable calculations (magnetization, entanglement entropy, projectors, etc.).
  - `fileIO.jl`: Trajectory and simulation data I/O, file naming, and data management.
  - `analysis.jl`: Analysis tools (swap operators, projectors, etc.).
  - `mpiSimulation.jl`: MPI-based parallel simulation orchestration.
- `test/` — Test scripts and regression tests for different models and simulation setups.
- `Project.toml`/`Manifest.toml` — Julia package/project dependencies and environment.
- `README.md` — Brief project description.

## Main Concepts
- **Simulation**: Top-level object containing all parameters and circuits for a simulation run.
- **Circuit**: Encapsulates a specific quantum circuit configuration (model, size, boundary, etc.).
- **Trajectory**: Represents a single quantum trajectory (state evolution, observables, etc.).
- **Projectors/Operators**: Constructed for each model and used in time evolution and measurement.
- **Observables**: Quantities measured during/after simulation (magnetization, entanglement, etc.).

## Coding Standards
- Use clear, descriptive function and variable names.
- Modularize code by physical or logical functionality.
- Use Julia's type system for clarity and performance (e.g., abstract types for trajectories).
- Prefer immutable structs for configuration, mutable for evolving state.
- Document all public functions and types with docstrings.
- Use `export` to define the public API in each module.
- Avoid hardcoding paths; use `joinpath` and configuration parameters.
- Use sparse matrices for large Hilbert spaces.
- Write tests for all major features in `test/`.

## Domain Knowledge
- Models supported: Fredkin, Motzkin, Biquadratic, SU(2), etc., for spin-1/2 and spin-1 chains.
- Supports both open (OBC) and periodic (PBC) boundary conditions.
- Parallelization via MPI for large-scale simulations.
- Observables and entanglement are central analysis targets.
- Initial states can be random, product, or highly entangled (Dicke, anomalous, etc.).

## Extending the Codebase
- Add new models by defining new circuit/projector construction and trajectory types.
- Add new observables by extending `observables.jl`.
- Add new initial states in `initialStates.jl`.
- Ensure new features are tested in `test/`.

## Preferences
- Use Julia 1.10+ and compatible package versions (see `Project.toml`).
- Use MPI for parallel runs; ensure code is robust to single- and multi-process execution.
- Keep I/O efficient by batching or reducing frequency of disk writes.
- Prefer clarity and maintainability over premature optimization.

## Source File Function Map

- **MonitoredSpinChains.jl**: Main module file. Imports dependencies, exports API, and includes all submodules.

- **struct.jl**: 
  - Defines core data structures: `Simulation`, `Circuit`, `Trajectory`, and their subtypes.
  - Implements utility functions for hashing, combining, and manipulating observables.

- **constants.jl**: 
  - Declares physical and mathematical constants (Pauli matrices, projectors, etc.).
  - Defines tensor product and identity matrix helpers.

- **initialStates.jl**: 
  - Functions to generate initial quantum states: `rand_spinhalf_im`, `rand_spinone_im`, `rand_spinhalf_real`, `rand_spinone_real`, `neelState`, `anomalous_ground_state`, `flat_spin1`, `generalized_dicke`, etc.

- **initialize.jl**: 
  - Projector construction for each model: `get_proj_fredkin`, `get_proj_motzkin`, `get_proj_su2`, `get_proj_aklt`, etc.
  - Functions for simulation and circuit setup: `create_simulation`, `get_projectors`, `get_trajectories_from_simulation`, `get_trajectories_from_circuit`, `determine_trajectory`, etc.
  - Parameter checking and feedback index helpers.

- **dynamics.jl**: 
  - Time evolution and measurement routines: `run_trajectory!`, `time_evolve!`, `thermalize!`, `time_step!`, `meas!`, `correct!`, etc.
  - Model-specific feedback/correction implementations.

- **observables.jl**: 
  - Observable calculations: `get_observables!`, `get_observables`, `total_projector`, `entanglement_entropy_general`, `magnetization`, `magnetizationX`, etc.
  - Analytical formulas for entanglement and related quantities.

- **fileIO.jl**: 
  - Trajectory and simulation data I/O: `save_trajectory`, `move_to_computed_folder`, `save_traj`, `load_existing_trajectory_data!`, etc.
  - File naming and data management helpers: `trajectory_to_filename`, `circuit_to_filename`.

- **analysis.jl**: 
  - Analysis tools: `swap_1_L`, `swap_spin1_sites`, `swaps`, `symmetric_projector`, etc.

- **mpiSimulation.jl**: 
  - MPI-based parallel simulation orchestration: `simulate`, `simulate_many_traj`, and RAM estimation helpers.

- **test/**: 
  - Test scripts for different models and simulation setups: `fs1.jl`, `full_sim.jl`, `normalization.jl`, `runtests.jl`.