#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstdio>
#include <time.h>
#include <cstdlib>
#include <cstring>

#include "../config.h"
#include "../structures/tupleList.h"

typedef enum
{
    IMMUNE = 0,
    INFECTED = 1,
    SUSCEPTIBLE = 2,
    DEAD = 3
} State;

typedef struct
{
    int x, y;
    State state;
} PersonReport;

/**
 * @brief Initializes the curand states for random number generation on the GPU.
 *
 * @param states Pointer to the array of curand states
 * @param seed Seed for the random number generator
 */
__global__ void init_curand_kernel(curandStatePhilox4_32_10_t *states, unsigned long long seed);

/**
 * @brief CPU function to get random positions that respect the occupancy constraints.
 *
 * @param h_x Pointer to the x-coordinates array
 * @param h_y Pointer to the y-coordinates array
 */
void gen_random_coords(int *h_x, int *h_y);

/**
 * @brief Calculates the elapsed time in milliseconds between two timespec values.
 *
 * @param start The starting timespec value.
 * @param end The ending timespec value.
 * @return The elapsed time in milliseconds as a long integer.
 */
long get_time_in_ms(struct timespec start, struct timespec end);

/**
 * @brief Returns the current memory usage in megabytes.
 *
 * @return The memory usage in MB.
 */
double get_memory_usage_mb();

/**
 * @brief Logs the current memory usage with a label.
 *
 * @param label The label to associate with the memory usage log.
 */
void log_memory_usage(const char *label);

/**
 * @brief Prints debug information about the simulation state.
 *
 * @param phase The current phase or label for the debug output.
 * @param d_x Device pointer to the x-coordinates array.
 * @param d_y Device pointer to the y-coordinates array.
 * @param d_incub Device pointer to the incubation array.
 * @param d_susc Device pointer to the susceptibility array.
 * @param d_cellCount Device pointer to the cell count array.
 */
void debugState(const char *phase,
                int *d_x, int *d_y,
                int *d_incub, float *d_susc,
                int *d_cellCount);

/**
 * @brief Copies device arrays back to host and writes reports
 *
 * @param d_x      device array of x coords (int[NP])
 * @param d_y      device array of y coords (int[NP])
 * @param d_incub  device array of incubation days (int[NP])
 * @param d_susc   device array of susceptibility (float[NP])
 * @param day      file index (0..ND-1)
 */
void save_population(const int *d_x,
                     const int *d_y,
                     const int *d_incub,
                     const float *d_susc,
                     int day);