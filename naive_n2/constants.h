#pragma once
#include <cmath>

constexpr double kb = 1.380649e-23;
constexpr double sigma = 3.405e-10;
constexpr double epsilon = 1.654e-21;
constexpr double m = 6.6335e-26;

constexpr double T0 = 80.0; // не называй T (конфликт с функцией temperature)

constexpr double r_cut = 2.5;
constexpr double r_cut2 = (r_cut * sigma) * (r_cut * sigma);

constexpr double invr2 = sigma * sigma / r_cut2;
constexpr double invr6 = invr2 * invr2 * invr2;
constexpr double invr12 = invr6 * invr6;

constexpr double U_shift = 4 * epsilon * (invr12 - invr6);
