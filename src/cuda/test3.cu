// simulation.cu
// A CUDA-based epidemic spread simulator using SoA layout + Person→Cell mapping via Thrust

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <string.h>

#include <cuda.h>
#include <curand_kernel.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/host_vector.h>

//------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------
long get_time_in_ms(struct timespec start, struct timespec end)
{
    return (end.tv_sec - start.tv_sec) * 1000 +
           (end.tv_nsec - start.tv_nsec) / 1000000;
}

double get_memory_usage_mb()
{
    FILE *file = fopen("/proc/self/status", "r");
    if (!file)
        return -1;
    char line[128];
    while (fgets(line, sizeof(line), file))
    {
        if (strncmp(line, "VmRSS:", 6) == 0)
        {
            long rss_kb;
            sscanf(line + 6, "%ld", &rss_kb);
            fclose(file);
            return rss_kb / 1024.0;
        }
    }
    fclose(file);
    return -1;
}

//------------------------------------------------------------------------------
// Simulation parameters
//------------------------------------------------------------------------------
#define W 100                               // grid width
#define H 100                               // grid height
#define MAXP_CELL 3                         // max people per cell
#define NP ((int)(0.1 * W * H * MAXP_CELL)) // total people
#define INFP 0.05f                          // initial infected %
#define IMM 0.01f                           // initial immune %
#define S_AVG 0.5f                          // avg susceptibility
#define ND 20                               // days to simulate
#define INCUBATION_DAYS 4                   // incubation period
#define BETA 0.8f                           // contagiousness factor
#define ITH 0.5f                            // infection threshold
#define IRD 1                               // infection radius
#define MU 0.6f                             // recovery prob

// State flags
#define STATE_DEAD 0
#define STATE_IMMUNE 1
#define STATE_SUSCEPTIBLE 2
#define STATE_INFECTED 3

//------------------------------------------------------------------------------
// Utility: linear cell index
//------------------------------------------------------------------------------
__device__ __host__ inline int cellIndex(int x, int y)
{
    return x + y * W;
}

//------------------------------------------------------------------------------
// A small 3-round Feistel on 32 bits (for deterministic pseudorandom mapping)
//------------------------------------------------------------------------------
__device__ inline uint32_t feistel32(uint32_t x, uint32_t seed)
{
    uint32_t L = x >> 16, R = x & 0xFFFF;
#pragma unroll
    for (int r = 0; r < 3; ++r)
    {
        uint32_t K = (seed >> (16 * r)) & 0xFFFF;
        uint32_t F = ((R ^ K) * 0x5bd1u) & 0xFFFF;
        uint32_t newL = R;
        uint32_t newR = L ^ F;
        L = newL;
        R = newR;
    }
    return (L << 16) | R;
}

//------------------------------------------------------------------------------
// Kernel 1: Initialization (deterministic roles, pseudorandom slot→cell)
//------------------------------------------------------------------------------
__global__ void init_kernel(
    unsigned long long seed,
    curandStatePhilox4_32_10_t *d_curandStates,
    int numImmune,
    int numInfected,
    int *d_x,
    int *d_y,
    int *d_cellIdx,
    uint8_t *d_newInf,
    uint8_t *d_state,
    float *d_susc,
    int *d_incub)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NP)
        return;

    // 1) Setup RNG for susceptibility noise
    curand_init(seed, tid, 0, &d_curandStates[tid]);

    // 2) Compute total slots
    const uint64_t M = (uint64_t)W * H * MAXP_CELL;

    // 3) Cycle‐walk Feistel to get a unique slot in [0..M)
    uint32_t v = (uint32_t)tid, y;
    do
    {
        y = feistel32(v, (uint32_t)seed);
        v = y;
    } while ((uint64_t)y >= M);
    uint64_t slot = y;

    // 4) Unpack slot → cell, x, y
    int cell = slot / MAXP_CELL;
    int px = cell % W;
    int py = cell / W;

    d_x[tid] = px;
    d_y[tid] = py;
    d_cellIdx[tid] = cell;
    d_newInf[tid] = 0;

    // 5) Deterministic role assignment by tid
    if (tid < numImmune)
    {
        d_state[tid] = STATE_IMMUNE;
        d_susc[tid] = 0.0f;
        d_incub[tid] = 0;
    }
    else if (tid < numImmune + numInfected)
    {
        d_state[tid] = STATE_INFECTED;
        d_incub[tid] = INCUBATION_DAYS + 1;
        d_susc[tid] = S_AVG + curand_normal(&d_curandStates[tid]) * 0.1f;
    }
    else
    {
        d_state[tid] = STATE_SUSCEPTIBLE;
        d_incub[tid] = 0;
        d_susc[tid] = S_AVG + curand_normal(&d_curandStates[tid]) * 0.1f;
    }
}

//------------------------------------------------------------------------------
// Kernel 2: Spread infection
//------------------------------------------------------------------------------
__global__ void infect_kernel(
    int *d_x, int *d_y,
    int *d_cellStart, int *d_cellCount,
    uint8_t *d_state, uint8_t *d_newInf,
    float *d_susc, int *d_incub)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NP || d_state[tid] != STATE_INFECTED)
        return;

    d_incub[tid]--;

    int x0 = d_x[tid], y0 = d_y[tid];
    for (int dy = -IRD; dy <= IRD; ++dy)
    {
        int y = y0 + dy;
        if (y < 0 || y >= H)
            continue;
        for (int dx = -IRD; dx <= IRD; ++dx)
        {
            int x = x0 + dx;
            if (x < 0 || x >= W)
                continue;

            int c = cellIndex(x, y);
            int start = d_cellStart[c];
            int count = d_cellCount[c];
            for (int i = start; i < start + count; ++i)
            {
                if (d_state[i] == STATE_SUSCEPTIBLE && !d_newInf[i])
                {
                    float infec = BETA * d_susc[i];
                    if (infec > ITH)
                        d_newInf[i] = 1;
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// Kernel 3: Recovery/death & commit new infections
//------------------------------------------------------------------------------
__global__ void status_kernel(
    curandStatePhilox4_32_10_t *d_curandStates,
    uint8_t *d_state, int *d_incub,
    float *d_susc, int *d_x, int *d_y,
    uint8_t *d_newInf)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NP)
        return;

    if (d_state[tid] == STATE_INFECTED && d_incub[tid] == 1)
    {
        float p = curand_uniform(&d_curandStates[tid]);
        if (p < MU)
        {
            if ((curand(&d_curandStates[tid]) & 1) == 0)
            {
                d_state[tid] = STATE_IMMUNE;
                d_susc[tid] = 0.0f;
            }
            else
            {
                d_state[tid] = STATE_SUSCEPTIBLE;
            }
            d_incub[tid] = 0;
        }
        else
        {
            d_state[tid] = STATE_DEAD;
            d_x[tid] = d_y[tid] = -1;
        }
    }
    if (d_newInf[tid])
    {
        d_state[tid] = STATE_INFECTED;
        d_incub[tid] = INCUBATION_DAYS + 1;
        d_newInf[tid] = 0;
    }
}

//------------------------------------------------------------------------------
// Kernel 4: Propose random movement
//------------------------------------------------------------------------------
__global__ void propose_move_kernel(
    curandStatePhilox4_32_10_t *d_curandStates,
    int *d_x, int *d_y, int *d_propX, int *d_propY,
    int *d_propCellIdx, uint8_t *d_state)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NP)
        return;

    if (d_state[tid] == STATE_DEAD)
    {
        d_propX[tid] = d_x[tid];
        d_propY[tid] = d_y[tid];
        d_propCellIdx[tid] = cellIndex(d_x[tid], d_y[tid]);
        return;
    }

    int x0 = d_x[tid], y0 = d_y[tid];
    int dx = (curand(&d_curandStates[tid]) % 3) - 1;
    int dy = (curand(&d_curandStates[tid]) % 3) - 1;
    int nx = x0 + dx, ny = y0 + dy;
    if (nx < 0 || nx >= W || ny < 0 || ny >= H)
    {
        nx = x0;
        ny = y0;
    }

    d_propX[tid] = nx;
    d_propY[tid] = ny;
    d_propCellIdx[tid] = cellIndex(nx, ny);
}

//------------------------------------------------------------------------------
// Kernel 5: Apply movement based on rank
//------------------------------------------------------------------------------
__global__ void apply_move_kernel(
    int *ranks,
    int *d_x, int *d_y, int *d_cellIdx,
    int *d_propX, int *d_propY, int *d_propCellIdx,
    uint8_t *d_state)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NP)
        return;

    if (d_state[tid] != STATE_DEAD && ranks[tid] < MAXP_CELL)
    {
        d_x[tid] = d_propX[tid];
        d_y[tid] = d_propY[tid];
        d_cellIdx[tid] = d_propCellIdx[tid];
    }
}

//------------------------------------------------------------------------------
// Build cell‐map via Thrust on host
//------------------------------------------------------------------------------
void rebuildCellMap(
    int *d_cellIdx,
    int *d_cellStart, int *d_cellCount,
    int *d_x, int *d_y,
    float *d_susc, int *d_incub,
    uint8_t *d_newInf, uint8_t *d_state)
{
    auto keys_begin = thrust::device_pointer_cast(d_cellIdx);
    auto keys_end = keys_begin + NP;

    // sort SoA by cellIdx
    thrust::sort_by_key(
        keys_begin, keys_end,
        thrust::make_zip_iterator(
            thrust::make_tuple(
                thrust::device_pointer_cast(d_x),
                thrust::device_pointer_cast(d_y),
                thrust::device_pointer_cast(d_susc),
                thrust::device_pointer_cast(d_incub),
                thrust::device_pointer_cast(d_newInf),
                thrust::device_pointer_cast(d_state))));

    // reduce_by_key → counts
    thrust::host_vector<int> h_uniqueKeys(NP), h_counts(NP);
    auto end_it = thrust::reduce_by_key(
        keys_begin, keys_end,
        thrust::make_constant_iterator(1),
        h_uniqueKeys.begin(),
        h_counts.begin());
    int unique_cells = end_it.first - h_uniqueKeys.begin();

    // compute offsets
    thrust::host_vector<int> h_offsets(unique_cells);
    int offset = 0;
    for (int i = 0; i < unique_cells; ++i)
    {
        h_offsets[i] = offset;
        offset += h_counts[i];
    }

    // fill host arrays
    std::vector<int> h_cellStart(W * H, -1), h_cellCount(W * H, 0);
    for (int i = 0; i < unique_cells; ++i)
    {
        int c = h_uniqueKeys[i];
        h_cellStart[c] = h_offsets[i];
        h_cellCount[c] = h_counts[i];
    }

    // copy to device
    cudaMemcpy(d_cellStart, h_cellStart.data(), W * H * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cellCount, h_cellCount.data(), W * H * sizeof(int), cudaMemcpyHostToDevice);
}

//------------------------------------------------------------------------------
// Entry point
//------------------------------------------------------------------------------
int main()
{
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    printf("Memory before: %.2f MB\n", get_memory_usage_mb());

    // exact counts
    int numImmune = (int)(IMM * NP);
    int numInfected = (int)(INFP * NP);

    // 1) allocate device arrays
    int *d_x, *d_y, *d_cellIdx, *d_propX, *d_propY, *d_propCellIdx;
    int *d_cellStart, *d_cellCount, *d_incub;
    float *d_susc;
    uint8_t *d_newInf, *d_state;
    curandStatePhilox4_32_10_t *d_curandStates;

    cudaMalloc(&d_x, NP * sizeof(int));
    cudaMalloc(&d_y, NP * sizeof(int));
    cudaMalloc(&d_susc, NP * sizeof(float));
    cudaMalloc(&d_incub, NP * sizeof(int));
    cudaMalloc(&d_newInf, NP * sizeof(uint8_t));
    cudaMalloc(&d_state, NP * sizeof(uint8_t));
    cudaMalloc(&d_cellIdx, NP * sizeof(int));
    cudaMalloc(&d_propX, NP * sizeof(int));
    cudaMalloc(&d_propY, NP * sizeof(int));
    cudaMalloc(&d_propCellIdx, NP * sizeof(int));
    cudaMalloc(&d_cellStart, W * H * sizeof(int));
    cudaMalloc(&d_cellCount, W * H * sizeof(int));
    cudaMalloc(&d_curandStates, NP * sizeof(curandStatePhilox4_32_10_t));

    thrust::device_vector<int> d_ranks(NP);

    int threads = 256;
    int blocks = (NP + threads - 1) / threads;

    // 2) initialize
    init_kernel<<<blocks, threads>>>(
        123456ULL,
        d_curandStates,
        numImmune,
        numInfected,
        d_x, d_y, d_cellIdx,
        d_newInf, d_state,
        d_susc, d_incub);
    cudaDeviceSynchronize();

    // 3) build initial cell‐map
    rebuildCellMap(
        d_cellIdx,
        d_cellStart, d_cellCount,
        d_x, d_y,
        d_susc, d_incub,
        d_newInf, d_state);

    // 4) main simulation loop
    for (int day = 0; day < ND; ++day)
    {
        infect_kernel<<<blocks, threads>>>(
            d_x, d_y,
            d_cellStart, d_cellCount,
            d_state, d_newInf,
            d_susc, d_incub);
        cudaDeviceSynchronize();

        status_kernel<<<blocks, threads>>>(
            d_curandStates,
            d_state, d_incub,
            d_susc, d_x, d_y,
            d_newInf);
        cudaDeviceSynchronize();

        propose_move_kernel<<<blocks, threads>>>(
            d_curandStates,
            d_x, d_y,
            d_propX, d_propY,
            d_propCellIdx,
            d_state);
        cudaDeviceSynchronize();

        // sort proposed moves
        {
            auto pkeys_begin = thrust::device_pointer_cast(d_propCellIdx);
            auto pkeys_end = pkeys_begin + NP;
            thrust::sort_by_key(
                pkeys_begin, pkeys_end,
                thrust::make_zip_iterator(
                    thrust::make_tuple(
                        thrust::device_pointer_cast(d_propX),
                        thrust::device_pointer_cast(d_propY),
                        thrust::device_pointer_cast(d_state))));
        }

        thrust::exclusive_scan_by_key(
            thrust::device_pointer_cast(d_propCellIdx),
            thrust::device_pointer_cast(d_propCellIdx) + NP,
            thrust::make_constant_iterator(1),
            d_ranks.begin());

        apply_move_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_ranks.data()),
            d_x, d_y, d_cellIdx,
            d_propX, d_propY, d_propCellIdx,
            d_state);
        cudaDeviceSynchronize();

        rebuildCellMap(
            d_cellIdx,
            d_cellStart, d_cellCount,
            d_x, d_y,
            d_susc, d_incub,
            d_newInf, d_state);
    }

    printf("Memory after:  %.2f MB\n", get_memory_usage_mb());

    // 5) cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_susc);
    cudaFree(d_incub);
    cudaFree(d_newInf);
    cudaFree(d_state);
    cudaFree(d_cellIdx);
    cudaFree(d_propX);
    cudaFree(d_propY);
    cudaFree(d_propCellIdx);
    cudaFree(d_cellStart);
    cudaFree(d_cellCount);
    cudaFree(d_curandStates);

    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Time: %ld ms\n", get_time_in_ms(start, end));
    return 0;
}