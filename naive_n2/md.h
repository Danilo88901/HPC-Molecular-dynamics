#pragma once
#include <random>   

// Lennard-Jones
double Lennard_Jones(int n, double sigma, double epsilon, double m,
    double r_cut2, double U_shift, double L,
    double* x, double* y, double* z,
    double* fx, double* fy, double* fz);

// Integrator
double integrate(int n, double sigma, double epsilon, double m,
    double dt, double r_cut2, double U_shift, double L,
    double* x, double* y, double* z,
    double* fx, double* fy, double* fz,
    double* vx, double* vy, double* vz,
    double* fx_old, double* fy_old, double* fz_old);

// Thermodynamics
double kinetic_energy(int n, double m,
    double* vx, double* vy, double* vz);

double temperature(int n, double m,
    double* vx, double* vy, double* vz);

// Init
double init_positions(int n, double* x, double* y, double* z,
    double density, double mass);

void init_velocities(int n,
    double* vx, double* vy, double* vz,
    std::mt19937& gen);
