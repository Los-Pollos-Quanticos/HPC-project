#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "../utils/utils_cuda.cuh"

__global__ void init_population_kernel(
    int *d_x, int *d_y, int *d_incub, float *d_susc,
    int *d_newInf, int *d_cellIdx,
    curandStatePhilox4_32_10_t *states,
    int num_immune, int num_infected, float stddev)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NP)
        return;

    d_cellIdx[i] = d_x[i] + d_y[i] * W;
    d_newInf[i] = 0;

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

__global__ void buildCellSlots(
    int *d_x, int *d_y, int *d_cellIdx,
    int *d_cellCount, int *d_cellSlots)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NP)
    {
        int idx = d_cellIdx[tid];
        if (idx >= 0)
        {
            int pos = atomicAdd(&d_cellCount[idx], 1);
            if (pos < MAXP_CELL)
            {
                d_cellSlots[idx * MAXP_CELL + pos] = tid;
            }
        }
    }
}

__device__ bool isImmune(int i, float *d_susc)
{
    return d_susc[i] == 0.0f;
}

__device__ bool isDead(int i, int *d_cellIdx)
{
    return d_cellIdx[i] < 0;
}

__device__ bool isInfected(int i, int *d_incub)
{
    return d_incub[i] > 0;
}

__global__ void infect_kernel(
    int *d_x, int *d_y, int *d_incub, float *d_susc,
    int *d_newInf, int *d_cellCount, int *d_cellSlots)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NP)
        return;
    if (!isInfected(tid, d_incub))
        return;

    d_incub[tid]--;

    int x0 = d_x[tid], y0 = d_y[tid];

    // explore Moore neighborhood
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

            int c = x + y * W;
            int count = d_cellCount[c];
            int start = c * MAXP_CELL;
            for (int idx = 0; idx < count; ++idx)
            {
                int i = d_cellSlots[start + idx];
                if (i == tid)
                    continue;

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
    int *d_newInf, int *d_cellIdx,
    curandStatePhilox4_32_10_t *d_curandStates)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NP)
        return;

    if (isInfected(tid, d_incub) && d_incub[tid] == 1)
    {
        float p = curand_uniform(&d_curandStates[tid]);
        if (p < MU)
        {
            if ((curand(&d_curandStates[tid]) & 1) == 0)
            {
                d_susc[tid] = 0.0f;
            }
            d_incub[tid] = 0;
        }
        else
        {
            int cellIdx = d_cellIdx[tid];
            atomicAdd(&d_cellCount[cellIdx], -1);

            d_cellIdx[tid] = -1;
            d_x[tid] = -1;
            d_y[tid] = -1;
            d_incub[tid] = 0;
        }
    }

    if (d_newInf[tid])
    {
        d_incub[tid] = INCUBATION_DAYS + 1;
        d_newInf[tid] = 0;
    }
}

// Random movement, checking per-cell capacity
__global__ void move_kernel(
    int *d_x, int *d_y, int *d_cellIdx, int *d_cellCount,
    curandStatePhilox4_32_10_t *states)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NP || isDead(tid, d_cellIdx))
        return;

    unsigned int r1 = curand(&states[tid]);
    unsigned int r2 = curand(&states[tid]);
    int dx = (int)(r1 % 3) - 1;
    int dy = (int)(r2 % 3) - 1;

    int oldX = d_x[tid], oldY = d_y[tid];
    int newX = oldX + dx, newY = oldY + dy;
    int oldCell = oldX + oldY * W;
    int newCell = newX + newY * W;

    if (newX < 0 || newX >= W || newY < 0 || newY >= H || newCell == oldCell)
        return;

    int prev = atomicAdd(&d_cellCount[newCell], 1);
    if (prev >= MAXP_CELL)
    {
        atomicSub(&d_cellCount[newCell], 1);
        return;
    }
    atomicSub(&d_cellCount[oldCell], 1);

    d_x[tid] = newX;
    d_y[tid] = newY;
    d_cellIdx[tid] = newCell;
}

int main(int argc, char **argv)
{
    bool debugEnabled = false;
    for (int i = 1; i < argc; ++i)
        if (strcmp(argv[i], "--debug") == 0)
            debugEnabled = true;

    printf("Simulation started\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Host buffers for initial coords -- removed after initialization
    int *h_x = (int *)malloc(NP * sizeof(int));
    int *h_y = (int *)malloc(NP * sizeof(int));

    // Device arrays - SOA
    int *d_x, *d_y, *d_incub, *d_newInf, *d_cellIdx;
    float *d_susc;
    int *d_cellCount, *d_cellSlots;
    curandStatePhilox4_32_10_t *d_curandStates;

    cudaMalloc(&d_x, NP * sizeof(int));
    cudaMalloc(&d_y, NP * sizeof(int));
    cudaMalloc(&d_incub, NP * sizeof(int));
    cudaMalloc(&d_newInf, NP * sizeof(int));
    cudaMalloc(&d_susc, NP * sizeof(float));
    cudaMalloc(&d_cellIdx, NP * sizeof(int));

    cudaMalloc(&d_cellCount, W * H * sizeof(int));
    cudaMalloc(&d_cellSlots, W * H * MAXP_CELL * sizeof(int));
    cudaMalloc(&d_curandStates, NP * sizeof(curandStatePhilox4_32_10_t));

    int threads = 256;
    int blocks = (NP + threads - 1) / threads;

    init_curand_kernel<<<blocks, threads>>>(d_curandStates, (unsigned long long)time(NULL));

    gen_random_coords(h_x, h_y);
    cudaMemcpy(d_x, h_x, NP * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, NP * sizeof(int), cudaMemcpyHostToDevice);
    free(h_x);
    free(h_y);

    float stddev = 0.1f;
    int num_immune = (int)(NP * IMM);
    int num_infected = (int)(NP * INFP);

    init_population_kernel<<<blocks, threads>>>(
        d_x, d_y, d_incub, d_susc, d_newInf, d_cellIdx, d_curandStates,
        num_immune, num_infected, stddev);

    cudaMemset(d_cellCount, 0, W * H * sizeof(int));
    buildCellSlots<<<blocks, threads>>>(
        d_x, d_y, d_cellIdx, d_cellCount, d_cellSlots);

    if (debugEnabled)
    {
        debugState("after initial bucket build",
                   d_x, d_y, d_cellIdx, d_incub, d_susc,
                   d_cellCount, nullptr);
    }

    for (int day = 0; day < ND; ++day)
    {
        infect_kernel<<<blocks, threads>>>(
            d_x, d_y, d_incub, d_susc, d_newInf,
            d_cellCount, d_cellSlots);
        status_kernel<<<blocks, threads>>>(
            d_x, d_y, d_incub, d_susc, d_newInf,
            d_cellIdx, d_curandStates);
        move_kernel<<<blocks, threads>>>(
            d_x, d_y, d_cellIdx, d_cellCount,
            d_curandStates);

        // Rebuild buckets for next day
        cudaMemset(d_cellCount, 0, W * H * sizeof(int));
        buildCellSlots<<<blocks, threads>>>(
            d_x, d_y, d_cellIdx, d_cellCount, d_cellSlots);

        if (debugEnabled)
        {
            char label[32];
            snprintf(label, sizeof(label), "after day %d", day);
            debugState(label,
                       d_x, d_y, d_cellIdx, d_incub, d_susc,
                       d_cellCount, nullptr);
        }
        cudaDeviceSynchronize();
    }

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_incub);
    cudaFree(d_susc);
    cudaFree(d_newInf);
    cudaFree(d_cellIdx);
    cudaFree(d_cellCount);
    cudaFree(d_cellSlots);
    cudaFree(d_curandStates);

    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Time: %ld ms\n", get_time_in_ms(start, end));
    return 0;
}
