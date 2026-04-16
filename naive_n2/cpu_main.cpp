#include <iostream>
#include <random>
#include <cstdlib>

#include "constants.h"
#include "md.h"
double dt=1e-15;
int n=1000;
int steps=10000;
int size=n*sizeof(double);



int  main() {
	std::random_device rd;
	std::mt19937 gen(rd());
	int size = n * sizeof(double);
	double* x = (double*)malloc(size);
	double* y = (double*)malloc(size);
	double* z = (double*)malloc(size);
	double density = 1400.0;//кг/м3
	double L = init_positions(n, x, y, z, density, m);
	double* vx = (double*)malloc(size);
	double* vy = (double*)malloc(size);
	double* vz = (double*)malloc(size);
	init_velocities(n, vx, vy, vz, gen);
	double* fx = (double*)calloc(n, sizeof(double));
	double* fy = (double*)calloc(n, sizeof(double));
	double* fz = (double*)calloc(n, sizeof(double));

	double* fx_old = (double*)calloc(n, sizeof(double));
	double* fy_old = (double*)calloc(n, sizeof(double));
	double* fz_old = (double*)calloc(n, sizeof(double));
	Lennard_Jones(n, sigma, epsilon, m, r_cut2, U_shift, L, x, y, z, fx, fy, fz);

	for (int i = 0; i < steps; i++) {
		double U = integrate(n, sigma, epsilon, m, dt, r_cut2, U_shift, L, x, y, z, fx, fy, fz, vx, vy, vz,
			fx_old, fy_old, fz_old);
		if (i % 1000 == 0) {
			double K = kinetic_energy(n, m, vx, vy, vz);
			double T = temperature(n, m, vx, vy, vz);
			double E = U + K;
			std::cout << "Total energy:" << E << " ";
			std::cout << "Kinetic energy:" << K << " ";
			std::cout << "Potential energy:" << U << " ";
			std::cout << "Temperature:" << T << " ";
			std::cout << std::endl;
		}
	}

}
