#pragma once
#include <random>

struct System {
    int N;

    double* x;
    double* y;
    double* z;

    double* vx;
    double* vy;
    double* vz;

    double* fx;
    double* fy;
    double* fz;

    int* type;
};

struct bond {
    int bonds;
    int* i;
    int* j;
    double* k;
    double* r0;
};


double kinetic_energy(System& arr);
double temperature(System& arr);

void init_velocities(System& arr, std::mt19937& gen);

double init_positions_mixture(System& arr, bond& bondss,
    double density_kg_m3, double fraction_ethene);
