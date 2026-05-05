#pragma once
#include <cuda_runtime.h>

__device__ inline double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull =
        (unsigned long long int*)address;

    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;

    do {
        assumed = old;
        double new_val = __longlong_as_double(assumed) + val;
        old = atomicCAS(address_as_ull,
            assumed,
            __double_as_longlong(new_val));
    } while (assumed != old);

    return __longlong_as_double(old);
}
