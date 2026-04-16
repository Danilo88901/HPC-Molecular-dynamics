#pragma once

#include <cmath>
#include "constants.h"
#include "parameters.h"

inline double r_cut2 = (r_cut * sigma) * (r_cut * sigma);

inline double invr2  = sigma * sigma / r_cut2;
inline double invr6  = invr2 * invr2 * invr2;
inline double invr12 = invr6 * invr6;

inline double U_shift = 4 * epsilon * (invr12 - invr6);

inline double v_std = std::sqrt(kb * T / m);
