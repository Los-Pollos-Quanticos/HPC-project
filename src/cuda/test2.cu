// simulation.cu
// A CUDA-based epidemic spread simulator using SoA layout + global Person→Cell mapping via Thrust
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <string.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/host_vector.h>

long get_time_in_ms(struct timespec start, struct timespec end)
{
    return (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
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
            return rss_kb / 1024.0; // Convert to MB
        }
    }
    fclose(file);
    return -1;
}

// Constants for simulation parameters
#define W 100                               // Width of the grid
#define H 100                               // Height of the grid
#define MAXP_CELL 3                         // Maximum number of people in a cell
#define NP ((int)(0.1 * W * H * MAXP_CELL)) // Number of people
#define INFP 0.05f                          // Initial percentage of infected persons
#define IMM 0.01f                           // Initial percentage of immune persons
#define S_AVG 0.5f                          // Susceptibility average
#define ND 20                               // Number of days in simulation
#define INCUBATION_DAYS 4                   // Incubation period in days
#define BETA 0.8f                           // Contagiousness factor
#define ITH 0.5f                            // Infection threshold
#define IRD 1                               // Infection radius (in cells)
#define MU 0.6f                             // Probability of recovery after infection

// State flags
#define STATE_DEAD 0
#define STATE_IMMUNE 1
#define STATE_SUSCEPTIBLE 2
#define STATE_INFECTED 3

// Device pointers for SoA arrays (length NP)
static int *d_x, *d_y;
static float *d_susc;
static int *d_incub;
static uint8_t *d_newInf, *d_state;
static int *d_cellIdx;         // current cell index per person
static int *d_propCellIdx;     // proposed cell index for movement
static int *d_propX, *d_propY; // proposed coords

// Device pointers for per-cell metadata (length W*H)
static int *d_cellStart, *d_cellCount;

// Device pointer for RNG state per thread
static curandStatePhilox4_32_10_t *d_curandStates;

// Host scratch for ranks and unique-key reductions
static thrust::device_vector<int> d_ranks;
static thrust::host_vector<int> h_uniqueKeys, h_counts, h_offsets;

// Utility: linear cell index
__device__ __host__ inline int cellIndex(int x, int y)
{
    return x + y * W;
}

//------------------------------------------------------------------------------
// Kernel 1: Initialization of positions, roles, and RNG
//------------------------------------------------------------------------------
__global__ void init_kernel(unsigned long long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NP)
        return;

    // Setup curand
    curand_init(seed, tid, 0, &d_curandStates[tid]);

    // Random slot among W*H*MAXP_CELL
    unsigned int total_slots = W * H * MAXP_CELL;
    unsigned int slot = curand(&d_curandStates[tid]) % total_slots;
    int cell = slot / MAXP_CELL;
    int x = cell % W;
    int y = cell / W;

    d_x[tid] = x;
    d_y[tid] = y;
    d_cellIdx[tid] = cell;
    d_newInf[tid] = 0;

    // Assign initial state by uniform random
    float r = curand_uniform(&d_curandStates[tid]);
    if (r < IMM)
    {
        d_state[tid] = STATE_IMMUNE;
        d_susc[tid] = 0.0f;
        d_incub[tid] = 0;
    }
    else if (r < IMM + INFP)
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
// Kernel 2: Infection spreading
//------------------------------------------------------------------------------
__global__ void infect_kernel()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NP)
        return;
    if (d_state[tid] != STATE_INFECTED)
        return;

    // decrement incubation
    d_incub[tid]--;

    int x0 = d_x[tid], y0 = d_y[tid];
    // loop over neighborhood
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
                    {
                        d_newInf[i] = 1;
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// Kernel 3: Recovery, death, and commit new infections
//------------------------------------------------------------------------------
__global__ void status_kernel()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NP)
        return;

    // If infected & last day
    if (d_state[tid] == STATE_INFECTED && d_incub[tid] == 1)
    {
        float p = curand_uniform(&d_curandStates[tid]);
        if (p < MU)
        {
            // recover
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
            // die
            d_state[tid] = STATE_DEAD;
            d_x[tid] = d_y[tid] = -1;
        }
    }

    // Commit new infections
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
__global__ void propose_move_kernel()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NP)
        return;

    // Dead stay put
    if (d_state[tid] == STATE_DEAD)
    {
        d_propX[tid] = d_x[tid];
        d_propY[tid] = d_y[tid];
        d_propCellIdx[tid] = d_cellIdx[tid];
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
// Kernel 5: Apply movement based on rank in cell
//------------------------------------------------------------------------------
__global__ void apply_move_kernel(int *ranks)
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
// Build d_cellStart and d_cellCount on host via Thrust reductions
//------------------------------------------------------------------------------
void rebuildCellMap()
{
    // Wrap raw pointer in thrust::device_ptr
    thrust::device_ptr<int> keys_begin(d_cellIdx), keys_end = keys_begin + NP;

    // 1) sort all SoA arrays by d_cellIdx
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

    // 2) reduce_by_key to count occupancy per active cell
    h_uniqueKeys.resize(W * H);
    h_counts.resize(W * H);
    auto end_it = thrust::reduce_by_key(
        keys_begin, keys_end,
        thrust::make_constant_iterator(1),
        h_uniqueKeys.begin(),
        h_counts.begin());
    int unique_cells = end_it.first - h_uniqueKeys.begin();

    // 3) build offsets on host
    h_offsets.resize(unique_cells);
    int offset = 0;
    for (int i = 0; i < unique_cells; ++i)
    {
        h_offsets[i] = offset;
        offset += h_counts[i];
    }

    // 4) create full-length host arrays with defaults
    std::vector<int> h_cellStart(W * H, -1), h_cellCount(W * H, 0);
    for (int i = 0; i < unique_cells; ++i)
    {
        int cell = h_uniqueKeys[i];
        h_cellStart[cell] = h_offsets[i];
        h_cellCount[cell] = h_counts[i];
    }

    // 5) copy down to device
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
    printf("Memory used before: %.2f MB\n", get_memory_usage_mb());

    // 1) Allocate all device arrays
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

    // Prepare device_vector for ranks
    d_ranks.resize(NP);

    // 2) Initialize
    int threads = 256;
    int blocks = (NP + threads - 1) / threads;
    init_kernel<<<blocks, threads>>>(123456ULL);
    cudaDeviceSynchronize();

    // 3) Build initial cell map
    rebuildCellMap();

    // 4) Main simulation loop
    for (int day = 0; day < ND; ++day)
    {
        // A) Infection spread
        infect_kernel<<<blocks, threads>>>();
        cudaDeviceSynchronize();

        // B) Recovery & commit
        status_kernel<<<blocks, threads>>>();
        cudaDeviceSynchronize();

        // C) Propose moves
        propose_move_kernel<<<blocks, threads>>>();
        cudaDeviceSynchronize();

        // D) Sort proposals by proposed cell
        {
            thrust::device_ptr<int> pkeys_begin(d_propCellIdx);
            thrust::device_ptr<int> pkeys_end = pkeys_begin + NP;

            thrust::sort_by_key(
                pkeys_begin, pkeys_end,
                thrust::make_zip_iterator(
                    thrust::make_tuple(
                        thrust::device_pointer_cast(d_propX),
                        thrust::device_pointer_cast(d_propY),
                        thrust::device_pointer_cast(d_state) // permute state so ranks align
                        )));
        }

        // E) Rank within each proposed cell
        thrust::exclusive_scan_by_key(
            thrust::device_pointer_cast(d_propCellIdx),
            thrust::device_pointer_cast(d_propCellIdx) + NP,
            thrust::make_constant_iterator(1),
            d_ranks.begin());

        // F) Apply moves based on rank
        apply_move_kernel<<<blocks, threads>>>(thrust::raw_pointer_cast(d_ranks.data()));
        cudaDeviceSynchronize();

        // G) Rebuild cell map for next iteration
        rebuildCellMap();
    }

    printf("Memory used after: %.2f MB\n", get_memory_usage_mb());

    // 5) Cleanup
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

    printf("Memory free : %.2f MB\n", get_memory_usage_mb());
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Time: %ld ms\n", get_time_in_ms(start, end));

    return 0;
}
