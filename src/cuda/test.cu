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

#define W 10000     // Width of the grid
#define H 10000     // Height of the grid
#define MAXP_CELL 3 // Maximum number of people in a cell
#define NP 10000000
#define INFP 0.5f         // Initial percentage of infected persons
#define IMM 0.1f          // Initial percentage of immune persons
#define S_AVG 0.5f        // Susceptibility average
#define ND 20             // Number of days in simulation
#define INCUBATION_DAYS 4 // Incubation period in days
#define BETA 0.8f         // Contagiousness factor
#define ITH 0.2f          // Infection threshold
#define IRD 1             // Infection radius (in cells)
#define MU 0.6f           // Probability of recovery after infection

void debugState(const char *phase,
                int *d_x, int *d_y,
                int *d_cellIdx,
                int *d_incub, float *d_susc,
                int *d_cellCount, int *d_cellStart)
{
    const int P_SAMPLE = NP; // show all persons
    const int G_PATCH = W;   // full grid width in this tiny example

    printf("===== DEBUG: %s =====\n", phase);

    // 1) copy down all person data
    int h_x[P_SAMPLE], h_y[P_SAMPLE], h_cellIdx[P_SAMPLE], h_incub[P_SAMPLE];
    float h_susc[P_SAMPLE];
    cudaMemcpy(h_x, d_x, sizeof(int) * P_SAMPLE, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y, d_y, sizeof(int) * P_SAMPLE, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cellIdx, d_cellIdx, sizeof(int) * P_SAMPLE, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_incub, d_incub, sizeof(int) * P_SAMPLE, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_susc, d_susc, sizeof(float) * P_SAMPLE, cudaMemcpyDeviceToHost);

    // 2) pretty‐print a table of each person
    printf(" Persons:\n");
    printf("  ID | CellIdx |   (x,y)   | IncubDays | Susc\n");
    printf(" ----+---------+-----------+-----------+------\n");
    for (int i = 0; i < P_SAMPLE; ++i)
    {
        printf(" %3d | %7d | (%2d,%2d) |     %2d    | %.2f\n",
               i,
               h_cellIdx[i],
               h_x[i],
               h_y[i],
               h_incub[i],
               h_susc[i]);
    }

    // 3) dump grid cell counts (top‐left G_PATCH×G_PATCH)
    int h_grid[G_PATCH * G_PATCH];
    cudaMemcpy(h_grid, d_cellCount,
               sizeof(int) * G_PATCH * G_PATCH,
               cudaMemcpyDeviceToHost);
    printf("\n Grid cellCount [0..%d)x[0..%d):\n", G_PATCH, G_PATCH);
    for (int y = 0; y < G_PATCH; ++y)
    {
        printf("  ");
        for (int x = 0; x < G_PATCH; ++x)
        {
            printf("%2d ", h_grid[x + y * G_PATCH]);
        }
        printf("\n");
    }

    // 4) overall status summary
    int dead = 0, infected = 0, immune = 0, susceptible = 0;
    for (int i = 0; i < P_SAMPLE; ++i)
    {
        if (h_x[i] < 0)
        {
            dead++;
        }
        else if (h_incub[i] > 0)
        {
            infected++;
        }
        else if (h_susc[i] == 0.0f)
        {
            immune++;
        }
        else
        {
            susceptible++;
        }
    }
    printf("\n Summary: dead=%d, infected=%d, immune=%d, susceptible=%d\n",
           dead, infected, immune, susceptible);

    printf("============================\n\n");
}

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
    int *d_cellCount,
    int *tmpKeys,
    int *tmpCounts)
{
    using thrust::constant_iterator;
    using thrust::device;
    using thrust::device_ptr;
    using thrust::make_tuple;
    using thrust::make_zip_iterator;

    // 1) pack values into a zip‐iterator
    auto zip_begin = make_zip_iterator(
        make_tuple(
            device_ptr<int>(d_cellIdx),
            device_ptr<int>(d_x),
            device_ptr<int>(d_y),
            device_ptr<float>(d_susc),
            device_ptr<int>(d_incub),
            device_ptr<int>(d_newInf)));
    auto zip_end = zip_begin + NP;

    // 2) partition out the “dead” cells
    auto live_end = thrust::stable_partition(device, zip_begin, zip_end, IsLive());
    int live_count = live_end - zip_begin;

    // 3) sort live cells by their cellIdx (keys are in-place in d_cellIdx[]; values are zipped)
    thrust::sort_by_key(
        device,
        d_cellIdx,
        d_cellIdx + live_count,
        make_zip_iterator(make_tuple(d_x, d_y, d_susc, d_incub, d_newInf)));

    // 4) run‐length encode (i.e. unique keys + counts)
    //    → writes unique keys into tmpKeys[0..numUnique)
    //    → writes counts    into tmpCounts[0..numUnique)
    int *endKeyPtr;
    int *endCountPtr;
    thrust::tie(endKeyPtr, endCountPtr) = thrust::reduce_by_key(
        device,
        d_cellIdx, d_cellIdx + live_count,
        constant_iterator<int>(1),
        tmpKeys,
        tmpCounts);
    int numUnique = endKeyPtr - tmpKeys;

    // 5) exclusive‐scan counts → directly into d_cellStart[key]
    thrust::exclusive_scan(
        device,
        tmpCounts, tmpCounts + numUnique,
        thrust::make_permutation_iterator(
            device_ptr<int>(d_cellStart),
            device_ptr<int>(tmpKeys)),
        0);

    // 6) scatter counts → directly into d_cellCount[key]
    thrust::scatter(
        device,
        tmpCounts, tmpCounts + numUnique,
        tmpKeys,
        d_cellCount);
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

int main(int argc, char **argv)
{
    bool debugEnabled = false;
    for (int i = 1; i < argc; ++i)
        if (strcmp(argv[i], "--debug") == 0)
            debugEnabled = true;

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

    int *tmpKeys;
    int *tmpCounts;
    cudaMalloc(&tmpKeys, NP * sizeof(int));
    cudaMalloc(&tmpCounts, NP * sizeof(int));

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

    rebuildCellMap(d_cellIdx, d_x, d_y, d_susc, d_incub, d_newInf, d_cellStart, d_cellCount, tmpKeys, tmpCounts);

    log_memory_usage("After rebuildCellMap");

    if (debugEnabled)
    {
        debugState("after rebuildCellMap",
                   d_x, d_y, d_cellIdx, d_incub, d_susc, d_cellCount,
                   d_cellStart);
    }

    for (int day = 0; day < ND; ++day)
    {
        infect_kernel<<<blocks, threads>>>(d_x, d_y, d_incub, d_susc, d_newInf, d_cellStart, d_cellCount);
        status_kernel<<<blocks, threads>>>(d_x, d_y, d_incub, d_susc, d_newInf, d_cellIdx, d_cellCount, d_curandStates);
        move_kernel<<<blocks, threads>>>(d_x, d_y, d_cellIdx, d_cellCount, d_curandStates);
        rebuildCellMap(d_cellIdx, d_x, d_y, d_susc, d_incub, d_newInf, d_cellStart, d_cellCount, tmpKeys, tmpCounts);
        log_memory_usage("After day");

        if (debugEnabled)
        {
            char label[32];
            snprintf(label, sizeof(label), "after day %d", day);
            debugState(label,
                       d_x, d_y, d_cellIdx, d_incub, d_susc, d_cellCount,
                       d_cellStart);
        }

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