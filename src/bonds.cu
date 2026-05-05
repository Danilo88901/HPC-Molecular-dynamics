#include "bonds.cuh"
#include <cmath>
#include <cuda_runtime.h>   
#include "cuda_utils.cuh"

__global__ void calculate_forces_bonds_gpu(double L, double L_inverse, const double* __restrict__ x, const double* __restrict__ y,
    const double* __restrict__ z, double* fx, double* fy, double* fz, int bonds, int* i,
    int* j, double* k, double* r0, double* U_total_energy) {
    
    __shared__ double U_total[256];
    int global = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    U_total[tx] = 0.0;



    if (global < bonds) {


        double r_cur = r0[global];
        double k_cur = k[global];

        int ii = i[global];
        int jj = j[global];

        double dx = x[jj] - x[ii];
        dx -= L * rint(dx * L_inverse);
        double dy = y[jj] - y[ii];
        dy -= L * rint(dy * L_inverse);
        double dz = z[jj] - z[ii];
        dz -= L * rint(dz * L_inverse);

        double r_inv = rsqrt(dx * dx + dy * dy + dz * dz);
        double r = 1.0 / r_inv;
        if (r < 1e-20)U_total[tx] = 0.0;
        else {
            double dr = (r - r_cur);
            U_total[tx] = 0.5 * k_cur * dr * dr;
            double F_scalar = k_cur * dr * r_inv;

            double fxx = F_scalar * dx;
            double fyy = F_scalar * dy;
            double fzz = F_scalar * dz;

            atomicAddDouble(&fx[ii], fxx);
            atomicAddDouble(&fy[ii], fyy);
            atomicAddDouble(&fz[ii], fzz);

            atomicAddDouble(&fx[jj], -fxx);
            atomicAddDouble(&fy[jj], -fyy);
            atomicAddDouble(&fz[jj], -fzz);
        }

    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tx < stride) {
            U_total[tx] += U_total[tx + stride];
        }
        __syncthreads();
    }
    if (tx == 0) {
        atomicAddDouble(&U_total_energy[0], U_total[tx]);
    }
}
