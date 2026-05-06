#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include <stdio.h>
#include <curand_kernel.h>
#include "params.hpp"
#include<random>
#include"system.hpp"
#include"cuda_utils.cuh"
#include "bonds.cuh"
#include "integrator_kernels.cuh"

__global__ void build_verlet_list(int n, double L, int max_neighbors, int* neighbor_count, int* neighbor_list,
	double* x, double* y, double* z, int* type,
	double r_cut2) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;


	if (i < n) {
        neighbor_count[i]=0;
		int count = 0;
		double xi = x[i];
		double yi = y[i];
		double zi = z[i];
		int type_i = type[i];
		for (int j = 0; j < n; j++) {
			if (i == j)continue;
			double dx = x[j] - xi;
			dx -= L * rint(dx / L);
			double dy = y[j] - yi;
			dy -= L * rint(dy / L);
			double dz = z[j] - zi;
			dz -= L * rint(dz / L);
			double r2 = dx * dx + dy * dy + dz * dz;

			if (r2 < r_cut2) {
				if (count < max_neighbors) {
					neighbor_list[i * max_neighbors + count] = j;
					count++;
				}
			}
		}
		neighbor_count[i] = count;
	}
}

__global__ void LJ_GPU(int n, double L, double L_inverse, double* x, double* y, double* z, double* fx, double* fy, double* fz, double sigma_CH3, double sigma_CH2,
    double sigma_mix, double epsilon_CH3, double epsilon_CH2, double epsilon_mix, double r_cut2_CH3, double r_cut2_CH2,
    double r_cut2_mix, double U_shift_CH3, double U_shift_CH2, double U_shift_mix, int* type, int* bond_partner, double* U_total_energy,
    int* neighbor_count, int* neighbor_list, int max_neighbors) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int txxx = threadIdx.x;
    double U_main = 0.0;
    __shared__ double U_total[256];
    U_total[txxx] = 0.0;
    if (i < n) {
        double xi = x[i];
        double yi = y[i];
        double zi = z[i];
        int type_i = type[i];
        double sigma = 0.0;
        double epsilon = 0.0;
        double U_shift = 0.0;
        double r_cut2 = 0.0;

        for (int k = 0; k < neighbor_count[i]; k++) {
            int j = neighbor_list[i * max_neighbors + k];
            if (j == bond_partner[i])continue;
            double dx = x[j] - xi;
            dx -= L * rint(dx * L_inverse);

            double dy = y[j] - yi;
            dy -= L * rint(dy * L_inverse);

            double dz = z[j] - zi;
            dz -= L * rint(dz * L_inverse);

            if (type_i == 0 && type[j] == 0) {
                sigma = sigma_CH3;
                epsilon = epsilon_CH3;
                U_shift = U_shift_CH3;
                r_cut2 = r_cut2_CH3;
            }
            else if (type_i == 1 && type[j] == 1) {
                sigma = sigma_CH2;
                epsilon = epsilon_CH2;
                U_shift = U_shift_CH2;
                r_cut2 = r_cut2_CH2;
            }
            else {
                sigma = sigma_mix;
                epsilon = epsilon_mix;
                U_shift = U_shift_mix;
                r_cut2 = r_cut2_mix;
            }
            double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 > r_cut2)continue;
            double inv_r2 = sigma * sigma / r2;
            double inv_r6 = inv_r2 * inv_r2 * inv_r2;
            double inv_r12 = inv_r6 * inv_r6;
            U_main += 4 * epsilon * (inv_r12 - inv_r6) - U_shift;
            double F_scalar = 24 * epsilon * (2 * inv_r12 - inv_r6) / r2;
            double fxx = -F_scalar * dx;
            double fyy = -F_scalar * dy;
            double fzz = -F_scalar * dz;
            fx[i] += fxx; fy[i] += fyy; fz[i] += fzz;
        }
    }


    if (i < n) {
        U_total[txxx] = U_main;
    }
    else {
        U_total[txxx] = 0.0;
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (txxx < stride) {
            U_total[txxx] += U_total[txxx + stride];
        }
        __syncthreads();
    }
    if (txxx == 0) {
        atomicAddDouble(&U_total_energy[0], 0.5 * U_total[txxx]);
    }

}
__global__ void check_rebuild_list(int n, double L, double* x, double* y, double* z,
    double* max, double* x_ref, double* y_ref, double* z_ref) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int tx = threadIdx.x;
    __shared__ double data[256];
    //if (i >= n)return;
    double disp2 = 0.0;

    if (i < n) {
        double dx = x[i] - x_ref[i];
        dx -= L * rint(dx / L);

        double dy = y[i] - y_ref[i];
        dy -= L * rint(dy / L);

        double dz = z[i] - z_ref[i];
        dz -= L * rint(dz / L);

        disp2 = dx * dx + dy * dy + dz * dz;
    }
    if (i < n) {
        data[tx] = disp2;
    }
    else {
        data[tx] = 0.0;
    }
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tx < stride) {
            if (data[tx] < data[tx + stride])data[tx] = data[tx + stride];
        }
        __syncthreads();
    }

    if (tx == 0) {
        atomicMaxDouble(&max[0], data[tx]);
    }

}



__global__ void setup_rand_kernel(curandState* state, unsigned long seed, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        // У каждого потока свой ID, чтобы последовательности не повторялись
        curand_init(seed, i, 0, &state[i]);
    }
}




#define cudaCheck(err) { \
    if (err != cudaSuccess) { \
        printf("CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}
int main() {
    std::random_device rd;
    std::mt19937 gen(42);
    System arr;
    arr.N = 10 * 10 * 10;
    int size = arr.N * sizeof(double);
    //POSITIONS
    arr.x = (double*)calloc(arr.N, sizeof(double));
    arr.y = (double*)calloc(arr.N, sizeof(double));
    arr.z = (double*)calloc(arr.N, sizeof(double));
    //VELOCITIES
    arr.vx = (double*)calloc(arr.N, sizeof(double));
    arr.vy = (double*)calloc(arr.N, sizeof(double));
    arr.vz = (double*)calloc(arr.N, sizeof(double));
    //ACCELERATION
    arr.fx = (double*)calloc(arr.N, sizeof(double));
    arr.fy = (double*)calloc(arr.N, sizeof(double));
    arr.fz = (double*)calloc(arr.N, sizeof(double));

    arr.type = (int*)calloc(arr.N, sizeof(int));

    bond bondss;
    bondss.bonds = arr.N / 2;
    bondss.i = (int*)malloc(bondss.bonds * sizeof(int));
    bondss.j = (int*)malloc(bondss.bonds * sizeof(int));
    bondss.k = (double*)malloc(bondss.bonds * sizeof(double));
    bondss.r0 = (double*)malloc(bondss.bonds * sizeof(double));
    double L = init_positions_mixture(arr, bondss, density, fraction_eten);
    init_velocities(arr, gen);
    double* d_x, * d_y, * d_z;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);
    cudaMalloc(&d_z, size);

    cudaMemcpy(d_x, arr.x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, arr.y, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, arr.z, size, cudaMemcpyHostToDevice);
    double* d_vx, * d_vy, * d_vz;
    cudaMalloc(&d_vx, size);
    cudaMalloc(&d_vy, size);
    cudaMalloc(&d_vz, size);

    cudaMemcpy(d_vx, arr.vx, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, arr.vy, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, arr.vz, size, cudaMemcpyHostToDevice);

    double* d_fx, * d_fy, * d_fz;
    cudaMalloc(&d_fx, size);
    cudaMalloc(&d_fy, size);
    cudaMalloc(&d_fz, size);


    cudaMemset(d_fx, 0, size);
    cudaMemset(d_fy, 0, size);
    cudaMemset(d_fz, 0, size);

    double* d_fx_old, * d_fy_old, * d_fz_old;

    cudaMalloc(&d_fx_old, size);
    cudaMalloc(&d_fy_old, size);
    cudaMalloc(&d_fz_old, size);
    cudaMemset(d_fx_old, 0, size);
    cudaMemset(d_fy_old, 0, size);
    cudaMemset(d_fz_old, 0, size);
    curandState* d_state;
    cudaMalloc(&d_state, arr.N * sizeof(curandState));


    // Запускаем один раз. seed может быть любым (например, time(0))
    setup_rand_kernel << <(arr.N + 255) / 256, 256 >> > (d_state, time(0), arr.N);
    cudaDeviceSynchronize();
    int* d_i, * d_j;
    cudaMalloc(&d_i, sizeof(int) * bondss.bonds);
    cudaMalloc(&d_j, sizeof(int) * bondss.bonds);

    cudaMemcpy(d_i, bondss.i, sizeof(int) * bondss.bonds, cudaMemcpyHostToDevice);
    cudaMemcpy(d_j, bondss.j, sizeof(int) * bondss.bonds, cudaMemcpyHostToDevice);

    double* d_k, * d_r0;
    cudaMalloc(&d_k, sizeof(double) * bondss.bonds);
    cudaMalloc(&d_r0, sizeof(double) * bondss.bonds);

    cudaMemcpy(d_k, bondss.k, sizeof(double) * bondss.bonds, cudaMemcpyHostToDevice);
    cudaMemcpy(d_r0, bondss.r0, sizeof(double) * bondss.bonds, cudaMemcpyHostToDevice);
    int* h_bond_partner = (int*)malloc(arr.N * sizeof(int));
    for (int i = 0; i < arr.N; i++) h_bond_partner[i] = -1; // Сначала никто не связан

    for (int k = 0; k < bondss.bonds; k++) {
        int u = bondss.i[k];
        int v = bondss.j[k];
        h_bond_partner[u] = v;
        h_bond_partner[v] = u;
    }
    int* d_type;
    cudaMalloc(&d_type, sizeof(int) * arr.N);
    cudaMemcpy(d_type, arr.type, sizeof(int) * arr.N, cudaMemcpyHostToDevice);

    int* d_h_bond_partner;
    cudaMalloc(&d_h_bond_partner, sizeof(int) * arr.N);
    cudaMemcpy(d_h_bond_partner, h_bond_partner, sizeof(int) * arr.N, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size_bonds = (bondss.bonds + block_size - 1) / block_size;
    int grid_size_LJ = (arr.N + block_size - 1) / block_size;
    double* d_U_total;
    cudaMalloc(&d_U_total, sizeof(double));
    cudaMemset(d_U_total, 0, sizeof(double));

    double U_total = 0.0;
    double p = nu * dt;
    double L_inverse = 1.0 / L;

    int* d_neighbor_count, * d_neighbor_list;
    int max_neighbors = 512;
    cudaMalloc(&d_neighbor_count, sizeof(int) * arr.N);
    cudaMalloc(&d_neighbor_list, sizeof(int) * arr.N * max_neighbors);

    cudaMemset(d_neighbor_count, 0, sizeof(int) * arr.N);
    cudaMemset(d_neighbor_list, 0, sizeof(int) * arr.N * max_neighbors);


    double* x_ref, * y_ref, * z_ref;
    cudaMalloc(&x_ref, size);
    cudaMalloc(&y_ref, size);
    cudaMalloc(&z_ref, size);

    cudaMemset(x_ref, 0, size);
    cudaMemset(y_ref, 0, size);
    cudaMemset(z_ref, 0, size);

    double r_skin = 0.3 * sigma_CH3;
    double r_cut2 = (2.5 * sigma_CH3 + r_skin) * (2.5 * sigma_CH3 + r_skin);
    build_verlet_list << <grid_size_LJ, block_size >> > (arr.N, L, max_neighbors, d_neighbor_count, d_neighbor_list,
        d_x, d_y, d_z, d_type, r_cut2);
    LJ_GPU << <grid_size_LJ, block_size >> > (arr.N, L, L_inverse, d_x, d_y, d_z, d_fx, d_fy, d_fz, sigma_CH3, sigma_CH2, sigma_mix,
        epsilon_CH3, epsilon_CH2, epsilon_mix, r_cut2_CH3, r_cut2_CH2, r_cut2_mix, U_shift_CH3, U_shift_CH2, U_shift_mix,
        d_type, d_h_bond_partner, d_U_total, d_neighbor_count, d_neighbor_list, max_neighbors);
    cudaMemcpy(x_ref, d_x, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(y_ref, d_y, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(z_ref, d_z, size, cudaMemcpyDeviceToDevice);
    double max_cpu = 0.0;
    double* max_gpu;
    cudaMalloc(&max_gpu, sizeof(double));
    cudaMemset(max_gpu, 0, sizeof(double));



    for (int step = 0; step < steps; step++) {
        cudaMemset(d_U_total, 0, sizeof(double));
        cudaMemset(max_gpu, 0, sizeof(double));
        //U_total = 0.0;

        change_pos << <grid_size_LJ, block_size >> > (arr.N, d_type, m_CH3, m_CH2, L, L_inverse, dt, d_x, d_y, d_z,
            d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, d_fx_old, d_fy_old, d_fz_old);
        cudaMemset(d_fx, 0, size);
        cudaMemset(d_fy, 0, size);
        cudaMemset(d_fz, 0, size);

        cudaCheck(cudaPeekAtLastError());
        calculate_forces_bonds_gpu << <grid_size_bonds, block_size >> > (L, L_inverse, d_x, d_y, d_z, d_fx, d_fy, d_fz, bondss.bonds,
            d_i, d_j, d_k, d_r0, d_U_total);
        cudaCheck(cudaPeekAtLastError());
        LJ_GPU << <grid_size_LJ, block_size >> > (arr.N, L, L_inverse, d_x, d_y, d_z, d_fx, d_fy, d_fz, sigma_CH3, sigma_CH2, sigma_mix,
            epsilon_CH3, epsilon_CH2, epsilon_mix, r_cut2_CH3, r_cut2_CH2, r_cut2_mix, U_shift_CH3, U_shift_CH2, U_shift_mix,
            d_type, d_h_bond_partner, d_U_total, d_neighbor_count, d_neighbor_list, max_neighbors);
        cudaCheck(cudaPeekAtLastError());
        change_velocities_and_thermostat << <grid_size_LJ, block_size >> > (d_state, arr.N, kb, T, d_type, m_CH3, m_CH2,
            L, dt, d_vx, d_vy, d_vz, d_fx, p, d_fy, d_fz, d_fx_old, d_fy_old, d_fz_old);
        cudaCheck(cudaPeekAtLastError());
        check_rebuild_list << <grid_size_LJ, block_size >> > (arr.N, L, d_x, d_y, d_z, max_gpu, x_ref, y_ref, z_ref);
        cudaMemcpy(&max_cpu, max_gpu, sizeof(double), cudaMemcpyDeviceToHost);
        if (max_cpu > (r_skin * 0.5) * (r_skin * 0.5)) {
            build_verlet_list << <grid_size_LJ, block_size >> > (arr.N, L, max_neighbors, d_neighbor_count, d_neighbor_list,
                d_x, d_y, d_z, d_type, r_cut2);
            cudaMemcpy(x_ref, d_x, size, cudaMemcpyDeviceToDevice);
            cudaMemcpy(y_ref, d_y, size, cudaMemcpyDeviceToDevice);
            cudaMemcpy(z_ref, d_z, size, cudaMemcpyDeviceToDevice);
        }
        if (step % 1000 == 0) {

            cudaMemcpy(&U_total, d_U_total, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(arr.vx, d_vx, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(arr.vy, d_vy, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(arr.vz, d_vz, size, cudaMemcpyDeviceToHost);

            double K = kinetic_energy(arr);
            double T = temperature(arr);
            double E = U_total + K;
            std::cout << "Total energy:" << E << " ";
            std::cout << "Kinetic energy:" << K << " ";
            std::cout << "Potential energy:" << U_total << " ";
            std::cout << "Temperature:" << T << " ";
            std::cout << std::endl;
        }
    }
}
