#pragma once
#include <cuda_runtime.h>
__global__ void calculate_forces_bonds_gpu(
    double L, double L_inverse,
    const double* x, const double* y, const double* z,
    double* fx, double* fy, double* fz,
    int bonds,
    int* i, int* j,
    double* k, double* r0,
    double* U_total_energy
);
