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
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/tuple.h>
#include <thrust/scatter.h>
#include <thrust/copy.h>
#include <thrust/partition.h>
#include <thrust/binary_search.h>

#define W 10000                           // Width of the grid
#define H 10000                           // Height of the grid
#define MAXP_CELL 3                       // Maximum number of people in a cell
#define NP ((int)(1 * W * H * MAXP_CELL)) // Number of people
// #define NP 7
#define INFP 0.5f         // Initial percentage of infected persons
#define IMM 0.1f          // Initial percentage of immune persons
#define S_AVG 0.5f        // Susceptibility average
#define ND 3              // Number of days in simulation
#define INCUBATION_DAYS 1 // Incubation period in days
#define BETA 0.8f         // Contagiousness factor
#define ITH 0.2f          // Infection threshold
#define IRD 1             // Infection radius (in cells)
#define MU 0.6f           // Probability of recovery after infection

typedef struct
{
    int x;
    int y;
} Tuple;

typedef struct
{
    Tuple *data;  // Array of tuples
    int size;     // Current number of valid tuples
    int capacity; // Maximum capacity of the array
} TList;

TList *createTList(int capacity)
{
    TList *arr = (TList *)malloc(sizeof(TList));
    arr->data = (Tuple *)malloc(capacity * sizeof(Tuple));
    arr->size = 0;
    arr->capacity = capacity;
    return arr;
}

void addTuple(TList *arr, int x, int y)
{
    if (arr->size >= arr->capacity)
    {
        printf("Array is full!\n");
        return;
    }
    arr->data[arr->size++] = (Tuple){x, y};
}

void removeTupleAt(TList *arr, int index)
{
    if (index < 0 || index >= arr->size)
    {
        printf("Invalid index\n");
        return;
    }
    arr->data[index] = arr->data[arr->size - 1];
    arr->size--;
}

int getRandomTupleIndex(TList *arr, Tuple *out)
{
    if (arr->size == 0)
        return -1;

    int idx = rand() % arr->size;
    *out = arr->data[idx];
    return idx;
}

void freeTList(TList *list)
{
    if (list == NULL)
        return;

    if (list->data != NULL)
        free(list->data);

    free(list);
}

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

void log_memory_usage(const char *label)
{
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);

    double free_mb = free_bytes / (1024.0 * 1024.0);
    double total_mb = total_bytes / (1024.0 * 1024.0);
    double used_mb = total_mb - free_mb;

    printf("---- %s ----\n", label);
    printf("Host memory usage: %.2f MB\n", get_memory_usage_mb());
    printf("Device memory usage: %.2f MB used / %.2f MB total (%.2f MB free)\n",
           used_mb, total_mb, free_mb);
    printf("---------------------------\n\n");
}

__global__ void init_curand_kernel(curandStatePhilox4_32_10_t *states, unsigned long long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NP)
    {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

void gen_random_coords(int *h_x, int *h_y)
{
    int *occupancy_map = (int *)malloc(W * H * sizeof(int));
    TList *available_coords = createTList(W * H);
    for (int x = 0; x < W; x++)
    {
        for (int y = 0; y < H; y++)
        {
            addTuple(available_coords, x, y);
            occupancy_map[x + y * W] = 0;
        }
    }

    srand(time(NULL));
    for (int i = 0; i < NP; ++i)
    {
        Tuple t;
        int idx = getRandomTupleIndex(available_coords, &t);

        h_x[i] = t.x;
        h_y[i] = t.y;
        occupancy_map[t.x + t.y * W]++;

        if (occupancy_map[t.x + t.y * W] == MAXP_CELL)
        {
            removeTupleAt(available_coords, idx);
        }
    }

    free(occupancy_map);
    freeTList(available_coords);
}

__global__ void init_population_kernel(int *d_x, int *d_y, int *d_incub, float *d_susc, int *d_newInf, int *d_cellIdx, curandStatePhilox4_32_10_t *states)
{
    float mean = S_AVG;
    float stddev = 0.1f;
    int num_immune = (int)(NP * IMM);
    int num_infected = (int)(NP * INFP);

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
        d_susc[i] = fminf(1.0f, fmaxf(1e-6f, mean + stddev * g));
    }
}

struct IsLive
{
    __host__ __device__ bool operator()(const thrust::tuple<int, int, int, float, int, int> &t) const
    {
        return thrust::get<0>(t) >= 0;
    }
};

void rebuildCellMap(
    int *d_cellIdx,
    int *d_x,
    int *d_y,
    float *d_susc,
    int *d_incub,
    int *d_newInf,
    int *d_cellStart,
    int *d_cellCount)
{
    const int maxCells = W * H;
    using thrust::device_ptr;
    using thrust::make_tuple;
    using thrust::make_zip_iterator;

    auto zip_begin = make_zip_iterator(
        make_tuple(
            device_ptr<int>(d_cellIdx),
            device_ptr<int>(d_x),
            device_ptr<int>(d_y),
            device_ptr<float>(d_susc),
            device_ptr<int>(d_incub),
            device_ptr<int>(d_newInf)));
    auto zip_end = zip_begin + NP;

    auto live_end = thrust::stable_partition(
        zip_begin, zip_end,
        IsLive());
    int live_count = live_end - zip_begin;

    auto keys_begin = device_ptr<int>(d_cellIdx);
    auto keys_live_end = keys_begin + live_count;
    auto vals_begin = make_zip_iterator(
        make_tuple(
            thrust::device_pointer_cast(d_x),
            thrust::device_pointer_cast(d_y),
            thrust::device_pointer_cast(d_susc),
            thrust::device_pointer_cast(d_incub),
            thrust::device_pointer_cast(d_newInf)));

    thrust::sort_by_key(
        keys_begin, keys_live_end,
        vals_begin);

    thrust::constant_iterator<int> ones(1);
    thrust::device_vector<int> d_uniqueKeys(maxCells), d_counts(maxCells);

    auto reduce_end = thrust::reduce_by_key(
        keys_begin, keys_live_end,
        ones,
        d_uniqueKeys.begin(),
        d_counts.begin());

    int unique_cells = reduce_end.first - d_uniqueKeys.begin();
    thrust::device_vector<int> d_offsets(unique_cells);

    thrust::exclusive_scan(
        d_counts.begin(),
        d_counts.begin() + unique_cells,
        d_offsets.begin(),
        0);

    thrust::device_vector<int> d_cellStart_vec(maxCells, -1);
    thrust::device_vector<int> d_cellCount_vec(maxCells, 0);

    thrust::scatter(
        d_offsets.begin(),
        d_offsets.begin() + unique_cells,
        d_uniqueKeys.begin(),
        d_cellStart_vec.begin());

    thrust::scatter(
        d_counts.begin(),
        d_counts.begin() + unique_cells,
        d_uniqueKeys.begin(),
        d_cellCount_vec.begin());

    thrust::copy(
        d_cellStart_vec.begin(), d_cellStart_vec.end(),
        device_ptr<int>(d_cellStart));

    thrust::copy(
        d_cellCount_vec.begin(), d_cellCount_vec.end(),
        device_ptr<int>(d_cellCount));
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
    int *d_x,
    int *d_y,
    int *d_incub,
    float *d_susc,
    int *d_newInf,
    int *d_cellStart,
    int *d_cellCount)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NP)
        return;

    if (!isInfected(tid, d_incub))
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

            int c = x + y * W;
            int start = d_cellStart[c];
            int count = d_cellCount[c];
            for (int i = start; i < start + count; ++i)
            {
                if (i == tid)
                    continue;

                if (!isInfected(i, d_incub) && !isImmune(i, d_susc))
                {
                    float infec = BETA * d_susc[i];
                    if (infec > ITH)
                    {
                        // If d_newInf[i] is 0, it sets it to 1 and returns 0 → this thread succeeds
                        atomicCAS(&d_newInf[i], 0, 1);
                    }
                }
            }
        }
    }
}

__global__ void status_kernel(
    int *d_x,
    int *d_y,
    int *d_incub,
    float *d_susc,
    int *d_newInf,
    int *d_cellIdx,
    int *d_cellCount,
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
            // recover
            if ((curand(&d_curandStates[tid]) & 1) == 0)
            {
                d_susc[tid] = 0.0f;
            }

            d_incub[tid] = 0;
        }
        else
        {
            int cellPos = d_cellIdx[tid];
            if (cellPos >= 0)
            {
                atomicSub(&d_cellCount[cellPos], 1);
            }
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

__global__ void move_kernel(
    int *d_x,
    int *d_y,
    int *d_cellIdx,
    int *d_cellCount,
    curandStatePhilox4_32_10_t *states)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NP)
        return;

    if (isDead(tid, d_cellIdx))
        return;

    unsigned int r1 = curand(&states[tid]);
    unsigned int r2 = curand(&states[tid]);
    int dx = (int)(r1 % 3) - 1;
    int dy = (int)(r2 % 3) - 1;

    int oldX = d_x[tid], oldY = d_y[tid];
    int newX = oldX + dx, newY = oldY + dy;

    if (newX < 0 || newX >= W || newY < 0 || newY >= H)
        return;
    int oldCell = oldX + oldY * W;
    int newCell = newX + newY * W;
    if (newCell == oldCell)
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

int main()
{
    printf("Simulation started\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    log_memory_usage("Start");

    int *h_x = (int *)malloc(NP * sizeof(int));
    int *h_y = (int *)malloc(NP * sizeof(int));

    int *d_x;
    int *d_y;
    int *d_incub;
    float *d_susc;
    int *d_newInf;
    cudaMalloc(&d_x, NP * sizeof(int));
    cudaMalloc(&d_y, NP * sizeof(int));
    cudaMalloc(&d_incub, NP * sizeof(int));
    cudaMalloc(&d_susc, NP * sizeof(float));
    cudaMalloc(&d_newInf, NP * sizeof(int));

    int *d_cellIdx;
    int *d_cellStart;
    int *d_cellCount;
    cudaMalloc(&d_cellIdx, NP * sizeof(int));
    cudaMalloc(&d_cellStart, W * H * sizeof(int));
    cudaMalloc(&d_cellCount, W * H * sizeof(int));

    int threads = 256;
    int blocks = (NP + threads - 1) / threads;

    curandStatePhilox4_32_10_t *d_curandStates;
    cudaMalloc(&d_curandStates, NP * sizeof(curandStatePhilox4_32_10_t));
    init_curand_kernel<<<blocks, threads>>>(d_curandStates, (unsigned long long)time(NULL));

    gen_random_coords(h_x, h_y);
    cudaMemcpy(d_x, h_x, NP * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, NP * sizeof(int), cudaMemcpyHostToDevice);
    free(h_x);
    free(h_y);

    init_population_kernel<<<blocks, threads>>>(d_x, d_y, d_incub, d_susc, d_newInf, d_cellIdx, d_curandStates);

    log_memory_usage("After population init");

    rebuildCellMap(d_cellIdx, d_x, d_y, d_susc, d_incub, d_newInf, d_cellStart, d_cellCount);

    log_memory_usage("After rebuildCellMap");

    for (int day = 0; day < ND; ++day)
    {
        infect_kernel<<<blocks, threads>>>(d_x, d_y, d_incub, d_susc, d_newInf, d_cellStart, d_cellCount);
        status_kernel<<<blocks, threads>>>(d_x, d_y, d_incub, d_susc, d_newInf, d_cellIdx, d_cellCount, d_curandStates);
        move_kernel<<<blocks, threads>>>(d_x, d_y, d_cellIdx, d_cellCount, d_curandStates);
        rebuildCellMap(d_cellIdx, d_x, d_y, d_susc, d_incub, d_newInf, d_cellStart, d_cellCount);
        log_memory_usage("After day");

        cudaDeviceSynchronize();
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_susc);
    cudaFree(d_incub);
    cudaFree(d_newInf);
    cudaFree(d_cellIdx);
    cudaFree(d_cellStart);
    cudaFree(d_cellCount);
    cudaFree(d_curandStates);

    log_memory_usage("After cleanup");

    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Time: %ld ms\n", get_time_in_ms(start, end));
}