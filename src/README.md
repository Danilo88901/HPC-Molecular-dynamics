This directory contains the implementation files of the molecular dynamics simulation.

It includes CUDA kernels, host-side logic, and entry points for different simulation approaches.

## Contents

- `system_init.cpp` – initialization of particle positions, velocities, and molecular configuration
- `bonds.cu` – CUDA implementation of bonded interactions (bond stretching forces)
- `integrator.cu` – CUDA implementation of the time integration scheme and thermostat logic
- `main_naive.cu` – full O(N²) Lennard-Jones simulation without spatial optimization
- `main_neighbor.cu` – optimized simulation using neighbor list / spatial decomposition (if implemented)

## Purpose

This directory contains all executable and GPU implementation code.

Different `main` files allow comparison between naive O(N²) and optimized neighbor-list-based approaches using the same physical model defined in `include/`.
