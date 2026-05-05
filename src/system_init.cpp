#include "params.hpp"
#include<random>
#include"system.hpp"
#include <cmath>
double kinetic_energy(System& arr) {
    double K = 0.0;
    for (int i = 0; i < arr.N; i++) {
        double m = 0.0;
        if (arr.type[i] == 0)m = m_CH3;
        else m = m_CH2;
        K += 0.5 * m * (arr.vx[i] * arr.vx[i] + arr.vy[i] * arr.vy[i] + arr.vz[i] * arr.vz[i]);
    }
    return K;
}
double temperature(System& arr) {
    double K = kinetic_energy(arr);
    int f = 3 * arr.N - 3;
    return 2 * K / (f * kb);
}

void init_velocities(System& arr, std::mt19937& gen) {
    for (int i = 0; i < arr.N; i++) {
        double m = 0.0;
        if (arr.type[i] == 0)m = m_CH3;
        else m = m_CH2;
        double v_stdd = sqrt(kb * T / m);
        std::normal_distribution<double>dist(0.0, v_stdd);
        arr.vx[i] = dist(gen);
        arr.vy[i] = dist(gen);
        arr.vz[i] = dist(gen);
    }
    double vx_sum = 0.0;
    double vy_sum = 0.0;
    double vz_sum = 0.0;
    for (int i = 0; i < arr.N; i++) {
        vx_sum += arr.vx[i];
        vy_sum += arr.vy[i];
        vz_sum += arr.vz[i];
    }
    vx_sum /= arr.N;
    vy_sum /= arr.N;
    vz_sum /= arr.N;
    for (int i = 0; i < arr.N; i++) {
        arr.vx[i] -= vx_sum;
        arr.vy[i] -= vy_sum;
        arr.vz[i] -= vz_sum;
    }
}








double init_positions_mixture(System& arr, bond& bondss,
    double density_kg_m3, double fraction_ethene) {

    double m_mol_etan = 2 * m_CH3;
    double m_mol_ethene = 2 * m_CH2;
    double avg_m_mol = (1.0 - fraction_ethene) * m_mol_etan + fraction_ethene * m_mol_ethene;

    int n_mol = arr.N / 2;
    double total_mass = n_mol * avg_m_mol;
    double V = total_mass / density_kg_m3;
    double L = std::cbrt(V);

    int n_side = (int)std::ceil(std::cbrt((double)n_mol));
    double spacing = L / n_side;

    std::mt19937 gen(1234567);
    std::uniform_real_distribution<double> jitter(-0.02 * spacing, 0.02 * spacing);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::uniform_real_distribution<double> angle(-1.0, 1.0);

    int mol_idx = 0;
    for (int i = 0; i < n_side && mol_idx < n_mol; ++i) {
        for (int j = 0; j < n_side && mol_idx < n_mol; ++j) {
            for (int k = 0; k < n_side && mol_idx < n_mol; ++k) {
                int a1 = mol_idx * 2;
                int a2 = mol_idx * 2 + 1;

                
                bool is_ethene = (dist(gen) < fraction_ethene);
                double r0_current = is_ethene ? 1.33e-10 : 1.54e-10;
                double k_current = is_ethene ? 750.0 : 500.0;

                arr.type[a1] = is_ethene ? 1 : 0;
                arr.type[a2] = is_ethene ? 1 : 0;

                double cx = (i + 0.5) * spacing + jitter(gen);
                double cy = (j + 0.5) * spacing + jitter(gen);
                double cz = (k + 0.5) * spacing + jitter(gen);

              
                double ux = angle(gen), uy = angle(gen), uz = angle(gen);
                double inv_len = 1.0 / std::sqrt(ux * ux + uy * uy + uz * uz);
                ux *= inv_len; uy *= inv_len; uz *= inv_len;

               
                arr.x[a1] = cx - 0.5 * r0_current * ux;
                arr.y[a1] = cy - 0.5 * r0_current * uy;
                arr.z[a1] = cz - 0.5 * r0_current * uz;

                arr.x[a2] = cx + 0.5 * r0_current * ux;
                arr.y[a2] = cy + 0.5 * r0_current * uy;
                arr.z[a2] = cz + 0.5 * r0_current * uz;

                
                bondss.i[mol_idx] = a1;
                bondss.j[mol_idx] = a2;
                bondss.r0[mol_idx] = r0_current;
                bondss.k[mol_idx] = k_current;

                mol_idx++;
            }
        }
    }
    return L;
}
