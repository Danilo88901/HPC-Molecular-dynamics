#include "integrator_kernels.cuh"
#include <curand_kernel.h>
#include <cmath>


__global__ void change_pos(int n, int* type, double m_CH3, double m_CH2, double L, double L_inverse,
    double dt, double* x, double* y, double* z,
    double* vx, double* vy, double* vz, double* fx,
    double* fy, double* fz, double* fx_old, double* fy_old, double* fz_old) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        fx_old[i] = fx[i];
        fy_old[i] = fy[i];
        fz_old[i] = fz[i];

        double m = (type[i] == 0) ? m_CH3 : m_CH2;
        double m_inverse = 1.0 / m;
        x[i] += vx[i] * dt + 0.5 * fx[i] * m_inverse * dt * dt;
        x[i] -= L * rint(x[i] * L_inverse);
        y[i] += vy[i] * dt + 0.5 * fy[i] * m_inverse * dt * dt;
        y[i] -= L * rint(y[i] * L_inverse);
        z[i] += vz[i] * dt + 0.5 * fz[i] * m_inverse * dt * dt;
        z[i] -= L * rint(z[i] * L_inverse);

    }
}

__global__ void change_velocities_and_thermostat(curandState* state, int n, double kb, double T,
    int* type, double m_CH3, double m_CH2, double L, double dt,
    double* vx, double* vy, double* vz, double* fx, double p,
    double* fy, double* fz, double* fx_old, double* fy_old, double* fz_old) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        double m = (type[i] == 0) ? m_CH3 : m_CH2;
        double m_inverse = 1.0 / m;
        vx[i] += 0.5 * (fx[i] + fx_old[i]) * m_inverse * dt;
        vy[i] += 0.5 * (fy[i] + fy_old[i]) * m_inverse * dt;
        vz[i] += 0.5 * (fz[i] + fz_old[i]) * m_inverse * dt;

        curandState local_state = state[i];

        double rand_u = curand_uniform_double(&local_state);

        if (rand_u < p) {
            double v_std_invr = rsqrt(kb * T / m);
            double v_std = 1.0 / v_std_invr;
            vx[i] = curand_normal_double(&local_state) * v_std;
            vy[i] = curand_normal_double(&local_state) * v_std;
            vz[i] = curand_normal_double(&local_state) * v_std;
        }
        state[i] = local_state;


    }
}
