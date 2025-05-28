#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <curand_kernel.h>
#include "../utils/utils_cuda.cuh"

__global__ void init_population_kernel(
    int *d_x, int *d_y, int *d_incub, float *d_susc,
    int *d_newInf, int *d_slotIndex,
    curandStatePhilox4_32_10_t *states,
    int num_immune, int num_infected, float stddev)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NP)
    {
        return;
    }

    d_newInf[i] = 0;
    // mark slotIndex invalid as default value
    d_slotIndex[i] = -1;

    if (i < num_immune)
    {
        d_incub[i] = 0;
        d_susc[i] = 0.0f;
    }
    else
    {
        d_incub[i] = (i < num_immune + num_infected) ? (INCUBATION_DAYS + 1) : 0;
        float g = curand_normal(&states[i]);
        d_susc[i] = fminf(1.0f, fmaxf(1e-6f, S_AVG + stddev * g));
    }
}

/**
 * Initializes the count, cell slots, and slot index structures
 *
 * @param d_x Array of x coordinates of particles
 * @param d_y Array of y coordinates of particles
 * @param d_cellCount Array to hold the count of particles in each cell
 * @param d_cellSlots Array to hold the slots of particles in each cell
 * @param d_slotIndex Array to hold the index of each particle in its cell
 */
__global__ void buildCellSlots(
    int *d_x, int *d_y, int *d_cellCount,
    int *d_cellSlots, int *d_slotIndex)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NP)
    {
        return;
    }

    int c = d_x[tid] + d_y[tid] * W;
    if (c < 0)
    {
        d_slotIndex[tid] = -1;
        return;
    }
    // append this particle to cell c
    int pos = atomicAdd(&d_cellCount[c], 1);
    if (pos < MAXP_CELL)
    {
        d_cellSlots[c * MAXP_CELL + pos] = tid;
        d_slotIndex[tid] = pos;
    }
}

// Helpers functions on the device
__device__ bool isImmune(int i, float *d_susc) { return d_susc[i] == 0.0f; }
__device__ bool isDead(int i, int *d_x, int *d_y) { return d_x[i] < 0 || d_y[i] < 0; }
__device__ bool isInfected(int i, int *d_incub) { return d_incub[i] > 0; }

__global__ void infect_kernel(
    int *d_x, int *d_y, int *d_incub, float *d_susc,
    int *d_newInf, int *d_cellCount, int *d_cellSlots)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NP || !isInfected(tid, d_incub))
    {
        return;
    }

    d_incub[tid]--;

    // Moore neighborhood infection
    int x0 = d_x[tid], y0 = d_y[tid];
    for (int dy = -IRD; dy <= IRD; ++dy)
    {
        int y = y0 + dy;
        if (y < 0 || y >= H)
        {
            continue;
        }
        for (int dx = -IRD; dx <= IRD; ++dx)
        {
            int x = x0 + dx;
            if (x < 0 || x >= W)
            {
                continue;
            }

            // get flat index for cell
            int c = x + y * W;
            // get number of persons in this cell
            int count = d_cellCount[c];
            // recover start of the slots in the d_cellSlots array
            int base = c * MAXP_CELL;
            for (int s = 0; s < count; ++s)
            {
                int i = d_cellSlots[base + s];
                if (i == tid)
                {
                    continue;
                }
                if (!isInfected(i, d_incub) && !isImmune(i, d_susc))
                {
                    float infec = BETA * d_susc[i];
                    if (infec > ITH)
                    {
                        atomicCAS(&d_newInf[i], 0, 1);
                    }
                }
            }
        }
    }
}

__global__ void status_kernel(
    int *d_x, int *d_y, int *d_incub, float *d_susc,
    int *d_newInf, int *d_cellCount,
    int *d_cellSlots, int *d_slotIndex,
    curandStatePhilox4_32_10_t *states)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NP)
        return;

    // if the infected person have completed ints incubation period (we initialize at the beginning with INCUBATION_DAYS + 1)
    if (isInfected(tid, d_incub) && d_incub[tid] == 1)
    {
        float p = curand_uniform(&states[tid]);
        bool dies = (p >= MU);
        if (!dies)
        {
            // recover, maybe become immune
            if ((curand(&states[tid]) & 1) == 0)
            {
                d_susc[tid] = 0.0f;
            }
            d_incub[tid] = 0;
        }
        else
        {
            // get flat index for cell
            int c = d_x[tid] + d_y[tid] * W;
            // get slot index
            int slot = d_slotIndex[tid];
            // remove form counts of the cell
            int oldCount = atomicSub(&d_cellCount[c], 1);
            // check if I was the last one in the cell
            int last = oldCount - 1;
            // if not the last, swap with the last one
            if (slot != last)
            {
                int other = d_cellSlots[c * MAXP_CELL + last];
                d_cellSlots[c * MAXP_CELL + slot] = other;
                d_slotIndex[other] = slot;
            }
            // mark dead
            d_slotIndex[tid] = -1;
            d_x[tid] = d_y[tid] = -1;
            d_incub[tid] = 0;
        }
    }

    // activate new infections
    if (d_newInf[tid])
    {
        d_incub[tid] = INCUBATION_DAYS + 1;
        d_newInf[tid] = 0;
    }
}

__global__ void move_kernel(
    int *d_x, int *d_y,
    int *d_cellCount, int *d_cellSlots, int *d_slotIndex,
    curandStatePhilox4_32_10_t *states)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NP || isDead(tid, d_x, d_y))
    {
        return;
    }

    unsigned int r1 = curand(&states[tid]);
    unsigned int r2 = curand(&states[tid]);
    int dx = int(r1 % 3) - 1;
    int dy = int(r2 % 3) - 1;

    int oldX = d_x[tid], oldY = d_y[tid];
    int newX = oldX + dx, newY = oldY + dy;
    if (newX < 0 || newX >= W || newY < 0 || newY >= H)
    {
        return;
    }

    int oldC = oldX + oldY * W;
    int newC = newX + newY * W;
    if (newC == oldC)
    {
        return;
    }

    int oldSlot = d_slotIndex[tid];

    // increments the count of the cell, if the cell was already full rollback, otherwise updates d_cellSlots and d_slotIndex
    int pos = atomicAdd(&d_cellCount[newC], 1);
    if (pos < MAXP_CELL)
    {
        // update d_cellSlots
        d_cellSlots[newC * MAXP_CELL + pos] = tid;
        // update d_slotIndex
        d_slotIndex[tid] = pos;
        // update position
        d_x[tid] = newX;
        d_y[tid] = newY;

        // remove from old cell
        int oldCount = atomicSub(&d_cellCount[oldC], 1);
        // check if I was the last one in the cell
        int last = oldCount - 1;
        // if not the last, swap with the last one
        if (oldSlot != last)
        {
            int other = d_cellSlots[oldC * MAXP_CELL + last];
            d_cellSlots[oldC * MAXP_CELL + oldSlot] = other;
            d_slotIndex[other] = oldSlot;
        }
    }
    else
    {
        // rollback
        atomicSub(&d_cellCount[newC], 1);
    }
}

int main(int argc, char **argv)
{
    bool debug = false;
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--debug"))
            debug = true;

    timespec ts, te;
    timespec ts_rm1, te_rm1;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    // Host coords only use to generate random coordinates
    int *h_x = (int *)malloc(NP * sizeof(int));
    int *h_y = (int *)malloc(NP * sizeof(int));
    gen_random_coords(h_x, h_y);

    // Device allocations
    int *d_x, *d_y, *d_incub, *d_newInf, *d_slotIndex;
    float *d_susc;
    int *d_cellCount, *d_cellSlots;
    curandStatePhilox4_32_10_t *d_states;

    cudaMalloc(&d_x, NP * sizeof(int));
    cudaMalloc(&d_y, NP * sizeof(int));
    cudaMalloc(&d_incub, NP * sizeof(int));
    cudaMalloc(&d_newInf, NP * sizeof(int));
    cudaMalloc(&d_susc, NP * sizeof(float));
    cudaMalloc(&d_slotIndex, NP * sizeof(int));

    cudaMalloc(&d_cellCount, W * H * sizeof(int));
    cudaMalloc(&d_cellSlots, W * H * MAXP_CELL * sizeof(int));
    cudaMalloc(&d_states, NP * sizeof(curandStatePhilox4_32_10_t));

    int threads = 256;
    int blocks = (NP + threads - 1) / threads;

    init_curand_kernel<<<blocks, threads>>>(d_states, (unsigned long long)time(nullptr));
    clock_gettime(CLOCK_MONOTONIC, &ts_rm1);
    cudaMemcpy(d_x, h_x, NP * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, NP * sizeof(int), cudaMemcpyHostToDevice);
    clock_gettime(CLOCK_MONOTONIC, &te_rm1);
    free(h_x);
    free(h_y);

    int num_immune = int(NP * IMM);
    int num_infected = int(NP * INFP);
    float stddev = 0.1f;

    init_population_kernel<<<blocks, threads>>>(
        d_x, d_y, d_incub, d_susc, d_newInf, d_slotIndex, d_states,
        num_immune, num_infected, stddev);

    cudaMemset(d_cellCount, 0, W * H * sizeof(int));
    buildCellSlots<<<blocks, threads>>>(
        d_x, d_y, d_cellCount,
        d_cellSlots, d_slotIndex);

    if (debug)
        debugState("initial build",
                   d_x, d_y, d_incub, d_susc,
                   d_cellCount, nullptr);

    // Main loop
    for (int day = 0; day < ND; ++day)
    {
        infect_kernel<<<blocks, threads>>>(
            d_x, d_y, d_incub, d_susc, d_newInf,
            d_cellCount, d_cellSlots);
        cudaDeviceSynchronize();
        status_kernel<<<blocks, threads>>>(
            d_x, d_y, d_incub, d_susc, d_newInf, d_cellCount,
            d_cellSlots, d_slotIndex, d_states);
        cudaDeviceSynchronize();
        move_kernel<<<blocks, threads>>>(
            d_x, d_y, d_cellCount, d_cellSlots, d_slotIndex, d_states);
        cudaDeviceSynchronize();

        if (debug)
        {
            char buf[32];
            snprintf(buf, 32, "after day %d", day);
            debugState(buf,
                       d_x, d_y, d_incub, d_susc,
                       d_cellCount, nullptr);
        }
    }

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_incub);
    cudaFree(d_newInf);
    cudaFree(d_susc);
    cudaFree(d_slotIndex);
    cudaFree(d_cellCount);
    cudaFree(d_cellSlots);
    cudaFree(d_states);

    clock_gettime(CLOCK_MONOTONIC, &te);
    int runtime = get_time_in_ms(ts, te);
    int runtime_memory_transfer = get_time_in_ms(ts_rm1, te_rm1);
    printf("RunTime: %ld ms\n", get_time_in_ms(ts, te));
    printf("RunTimeEvo: %ld ms\n", runtime - runtime_memory_transfer);
    return 0;
}
