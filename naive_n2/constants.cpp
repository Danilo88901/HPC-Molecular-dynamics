#include "constants.h"
#include <cmath>

const double kb = 1.380649e-23;
const double sigma = 3.405e-10;
const double epsilon = 1.654e-21;
const double m = 6.6335e-26;
const double T = 80.0;
const double v_std = sqrt(kb * T / m);
const doulbe M_PI=3.1415926535;
const double r_cut = 2.5 * sigma;
const double r_cut2 = r_cut * r_cut;

double inv_r2_cut = sigma * sigma / r_cut2;
double inv_r6_cut = inv_r2_cut * inv_r2_cut * inv_r2_cut;
double inv_r12_cut = inv_r6_cut * inv_r6_cut;

double U_shift = 4.0 * epsilon * (inv_r12_cut - inv_r6_cut)
