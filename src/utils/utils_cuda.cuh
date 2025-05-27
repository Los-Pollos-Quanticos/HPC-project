#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstdio>
#include <time.h>
#include <cstdlib>
#include <cstring>

#include "../config.h"
#include "../structures/tupleList.h"

__global__ void init_curand_kernel(curandStatePhilox4_32_10_t *states, unsigned long long seed);

void gen_random_coords(int *h_x, int *h_y);

long get_time_in_ms(struct timespec start, struct timespec end);

double get_memory_usage_mb();
void log_memory_usage(const char *label);

void debugState(const char *phase,
                int *d_x, int *d_y,
                int *d_incub, float *d_susc,
                int *d_cellCount, int *d_cellStart);
