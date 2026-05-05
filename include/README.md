# Include

This directory contains all shared header files used across the project.

These files define the core structures, physical parameters, and reusable CUDA utilities required for both simulation implementations (naive O(N²) , neighbor list versions and cell list versions).

## Contents

- `params.hpp` – physical constants, simulation parameters, and Lennard-Jones parameters
- `system.hpp` – system data structures (positions, velocities, forces, bonds)
- `cuda_utils.cuh` – CUDA helper functions (e.g., atomic operations for doubles)
- `bonds.cuh` – bond force kernel interface
- `integrator_kernels.cuh` – CUDA kernels for integration and thermostat

## Purpose

All files in this directory are shared between different simulation implementations to ensure consistency of physical models and reduce code duplication.
