#pragma once
#include <cmath>

// =====================
// Simulation parameters
// =====================
constexpr double dt = 0.5e-15;
constexpr double density = 500.0;      // kg/m^3
constexpr double fraction_eten = 0.5;

// =====================
// Physical constants
// =====================
constexpr double kb = 1.380649e-23;
constexpr double T = 160.0;
constexpr double nu = 1e13;

// =====================
// Masses
// =====================
constexpr double m_CH3 = 2.5e-26;
constexpr double m_CH2 = 2.329e-26;

// =====================
// LJ parameters CH3
// =====================
constexpr double sigma_CH3 = 3.75e-10;
constexpr double epsilon_CH3 = 98.0 * kb;
constexpr double r_cut_CH3 = 2.5 * sigma_CH3;
constexpr double r_cut2_CH3 = r_cut_CH3 * r_cut_CH3;

// =====================
// LJ parameters CH2
// =====================
constexpr double sigma_CH2 = 3.675e-10;
constexpr double epsilon_CH2 = 85.0 * kb;
constexpr double r_cut_CH2 = 2.5 * sigma_CH2;
constexpr double r_cut2_CH2 = r_cut_CH2 * r_cut_CH2;

// =====================
// Mixed LJ (?? constexpr ??-?? sqrt)
// =====================
const double sigma_mix = (sigma_CH3 + sigma_CH2) / 2.0;
const double epsilon_mix = std::sqrt(epsilon_CH3 * epsilon_CH2);
const double r_cut_mix = 2.5 * sigma_mix;
const double r_cut2_mix = r_cut_mix * r_cut_mix;

// =====================
// Energy shift CH3
// =====================
constexpr double invr2_CH3 = sigma_CH3 * sigma_CH3 / r_cut2_CH3;
constexpr double invr6_CH3 = invr2_CH3 * invr2_CH3 * invr2_CH3;
constexpr double invr12_CH3 = invr6_CH3 * invr6_CH3;
constexpr double U_shift_CH3 = 4 * epsilon_CH3 * (invr12_CH3 - invr6_CH3);

// =====================
// Energy shift CH2
// =====================
constexpr double invr2_CH2 = sigma_CH2 * sigma_CH2 / r_cut2_CH2;
constexpr double invr6_CH2 = invr2_CH2 * invr2_CH2 * invr2_CH2;
constexpr double invr12_CH2 = invr6_CH2 * invr6_CH2;
constexpr double U_shift_CH2 = 4 * epsilon_CH2 * (invr12_CH2 - invr6_CH2);

// =====================
// Energy shift MIX (?? constexpr)
// =====================
const double invr2_mix = sigma_mix * sigma_mix / r_cut2_mix;
const double invr6_mix = invr2_mix * invr2_mix * invr2_mix;
const double invr12_mix = invr6_mix * invr6_mix;
const double U_shift_mix = 4 * epsilon_mix * (invr12_mix - invr6_mix);
