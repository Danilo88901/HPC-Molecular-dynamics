#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void change_pos(int n, int* type, double m_CH3, double m_CH2, double L, double L_inverse,
    double dt, double* x, double* y, double* z,
    double* vx, double* vy, double* vz, double* fx,
    double* fy, double* fz, double* fx_old, double* fy_old, double* fz_old);

__global__ void change_velocities_and_thermostat(curandState* state, int n, double kb, double T,
    int* type, double m_CH3, double m_CH2, double L, double dt,
    double* vx, double* vy, double* vz, double* fx, double p,
    double* fy, double* fz, double* fx_old, double* fy_old, double* fz_old);
