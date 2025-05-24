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

#define W 4                                 // Width of the grid
#define H 4                                 // Height of the grid
#define MAXP_CELL 3                         // Maximum number of people in a cell
#define NP ((int)(0.4 * W * H * MAXP_CELL)) // Number of people
#define INFP 0.2f                           // Initial percentage of infected persons
#define IMM 0.1f                            // Initial percentage of immune persons
#define S_AVG 0.5f                          // Susceptibility average
#define ND 3                                // Number of days in simulation
#define INCUBATION_DAYS 1                   // Incubation period in days
#define BETA 0.8f                           // Contagiousness factor
#define ITH 0.2f                            // Infection threshold
#define IRD 1                               // Infection radius (in cells)
#define MU 0.6f                             // Probability of recovery after infection

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

float gaussian_random(float mean, float stddev)
{
    float u = ((float)rand() / RAND_MAX);
    float v = ((float)rand() / RAND_MAX);
    float s = mean + stddev * sqrt(-2.0f * log(u)) * cos(2.0f * M_PI * v);

    if (s < 0.0f)
    {
        return 0.0f;
    }

    if (s > 1.0f)
    {
        return 1.0f;
    }

    return s;
}

void init_population(
    int *h_x,
    int *h_y,
    int *h_incub,
    float *h_susc,
    int *h_newInf,
    int *h_cellIdx)
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

    int num_immune = (int)(NP * IMM);
    int num_infected = (int)(NP * INFP);

    srand(time(NULL));
    for (int i = 0; i < NP; ++i)
    {
        Tuple t;
        int idx = getRandomTupleIndex(available_coords, &t);
        printf("idx: %d, x: %d, y: %d\n", idx, t.x, t.y);

        h_x[i] = t.x;
        h_y[i] = t.y;
        h_newInf[i] = 0;
        h_cellIdx[i] = h_x[i] + h_y[i] * W;
        occupancy_map[t.x + t.y * W]++;

        if (i < num_immune)
        {
            h_susc[i] = 0.0f;
            h_incub[i] = 0;
        }
        else
        {
            // assign incubation period only to infected people
            h_incub[i] = (i < num_immune + num_infected) ? (INCUBATION_DAYS + 1) : 0;
            // assign susceptibility to both susceptible and infected people
            h_susc[i] = gaussian_random(S_AVG, 0.1f);
        }

        if (occupancy_map[t.x + t.y * W] == MAXP_CELL)
        {
            removeTupleAt(available_coords, idx);
        }
    }

    free(occupancy_map);
    freeTList(available_coords);
}

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
    using thrust::constant_iterator;
    using thrust::device_ptr;
    using thrust::device_vector;
    using thrust::make_tuple;
    using thrust::make_zip_iterator;

    device_ptr<int> keys_begin(d_cellIdx), keys_end = keys_begin + NP;

    thrust::sort_by_key(
        keys_begin, keys_end,
        thrust::make_zip_iterator(
            thrust::make_tuple(
                thrust::device_pointer_cast(d_x),
                thrust::device_pointer_cast(d_y),
                thrust::device_pointer_cast(d_susc),
                thrust::device_pointer_cast(d_incub),
                thrust::device_pointer_cast(d_newInf))));

    device_vector<int> d_uniqueKeys(maxCells);
    device_vector<int> d_counts(maxCells);

    auto reduce_end = thrust::reduce_by_key(
        keys_begin, keys_end,
        constant_iterator<int>(1),
        d_uniqueKeys.begin(),
        d_counts.begin());
    int unique_cells = reduce_end.first - d_uniqueKeys.begin();

    device_vector<int> d_offsets(unique_cells);
    thrust::exclusive_scan(
        d_counts.begin(),
        d_counts.begin() + unique_cells,
        d_offsets.begin(),
        0);

    device_vector<int> d_cellStart_vec(maxCells, -1);
    device_vector<int> d_cellCount_vec(maxCells, 0);

    thrust::scatter(
        d_offsets.begin(), d_offsets.begin() + unique_cells,
        d_uniqueKeys.begin(),
        d_cellStart_vec.begin());
    thrust::scatter(
        d_counts.begin(), d_counts.begin() + unique_cells,
        d_uniqueKeys.begin(),
        d_cellCount_vec.begin());

    thrust::copy(
        d_cellStart_vec.begin(),
        d_cellStart_vec.end(),
        device_ptr<int>(d_cellStart));
    thrust::copy(
        d_cellCount_vec.begin(),
        d_cellCount_vec.end(),
        device_ptr<int>(d_cellCount));
}

__global__ void init_curand_kernel(curandStatePhilox4_32_10_t *states, unsigned long long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NP)
    {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

__device__ bool isImmune(int i, float *d_susc)
{
    return d_susc[i] == 0.0f;
}

__device__ bool isDead(int i, int *d_x, int *d_y)
{
    return d_x[i] < 0 || d_y[i] < 0;
}

__device__ bool isInfected(int i, int *d_incub, int *d_x, int *d_y)
{
    return d_incub[i] > 0 && !isDead(i, d_x, d_y);
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

    if (!isInfected(tid, d_incub, d_x, d_y))
        return;

    // print information about the person
    printf("Person %d: (%d, %d), incub: %d, susc: %.2f, newInf: %d\n",
           tid, d_x[tid], d_y[tid], d_incub[tid], d_susc[tid], d_newInf[tid]);

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

            int c = x + y * W;
            int start = d_cellStart[c];
            int count = d_cellCount[c];
            for (int i = start; i < start + count; ++i)
            {
                if (i == tid)
                    continue;

                // print information about the person found in the cell and tell the person that is scanning it
                printf("Person %d: (%d, %d), scanned by person %d, incub: %d, susc: %.2f, newInf: %d\n",
                       i, d_x[i], d_y[i], tid, d_incub[i], d_susc[i], d_newInf[i]);
                if (!isInfected(i, d_incub, d_x, d_y) && !isImmune(i, d_susc))
                {
                    float infec = BETA * d_susc[i];
                    if (infec > ITH)
                    {
                        // If d_newInf[i] is 0, it sets it to 1 and returns 0 → this thread succeeds
                        if (atomicCAS(&d_newInf[i], 0, 1) == 0)
                        {
                            // This thread was the first to infect
                            printf("Person %d infected by person %d\n", i, tid);
                        }
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
    curandStatePhilox4_32_10_t *d_curandStates)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NP)
        return;

    // If infected & last day
    if (isInfected(tid, d_incub, d_x, d_y) && d_incub[tid] == 1)
    {
        float p = curand_uniform(&d_curandStates[tid]);
        if (p < MU)
        {
            // recover
            if ((curand(&d_curandStates[tid]) & 1) == 0)
            {
                d_susc[tid] = 0.0f;
                // print information about the person
                printf("Person %d recovered and is now immune\n", tid);
            }

            d_incub[tid] = 0;
            printf("Person %d recovered\n", tid);
        }
        else
        {
            d_x[tid] = d_y[tid] = -1;
            printf("Person %d died\n", tid);
        }
    }

    // Commit new infections
    if (d_newInf[tid])
    {
        d_incub[tid] = INCUBATION_DAYS + 1;
        d_newInf[tid] = 0;
        printf("Person %d is now infected\n", tid);
    }
}

int main()
{
    printf("Simulation started\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    log_memory_usage("Before host allocations");

    // Flattening of the People array in host
    int *h_x = (int *)malloc(NP * sizeof(int));
    int *h_y = (int *)malloc(NP * sizeof(int));
    int *h_incub = (int *)malloc(NP * sizeof(int));
    float *h_susc = (float *)malloc(NP * sizeof(float));
    int *h_newInf = (int *)calloc(NP, sizeof(int));

    int *h_cellIdx = (int *)malloc(NP * sizeof(int));

    log_memory_usage("After host allocations");

    // Flattening of the People array in device
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
    cudaMalloc(&d_cellIdx, NP * sizeof(int));

    int *d_cellStart; // tells me where the first person in a cell c is in the d_cellIdx array, -1 if empty
    int *d_cellCount; // tells me how many people are in the cell c, 0 if empty
    cudaMalloc(&d_cellStart, W * H * sizeof(int));
    cudaMalloc(&d_cellCount, W * H * sizeof(int));

    // use to manage the constraint on the number of people in a cell
    thrust::device_vector<int> d_ranks(NP);

    // Device pointer for RNG state per thread
    curandStatePhilox4_32_10_t *d_curandStates;
    cudaMalloc(&d_curandStates, NP * sizeof(curandStatePhilox4_32_10_t));

    log_memory_usage("After device allocations");

    init_population(
        h_x,
        h_y,
        h_incub,
        h_susc,
        h_newInf,
        h_cellIdx);

    log_memory_usage("After population init (host)");

    // print the situation of the arrays in a way that is really readable
    // like print the matrix and the people in it
    // and a list of people with their properties
    // and also a ledgend of how many people per state there are state are infected, immune, susceptible

    for (int i = 0; i < NP; ++i)
    {
        printf("Person %d: (%d, %d), incub: %d, susc: %.2f, newInf: %d\n",
               i, h_x[i], h_y[i], h_incub[i], h_susc[i], h_newInf[i]);
    }
    // show the matrix
    printf("Matrix:\n");
    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            int count = 0;
            for (int i = 0; i < NP; ++i)
            {
                if (h_x[i] == x && h_y[i] == y)
                {
                    count++;
                }
            }
            printf("%d ", count);
        }
        printf("\n");
    }

    // print the h_cellIdx array
    printf("Cell index:\n");
    for (int i = 0; i < NP; ++i)
    {
        printf("%d ", h_cellIdx[i]);
    }

    cudaMemcpy(d_x, h_x, NP * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, NP * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_incub, h_incub, NP * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_susc, h_susc, NP * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_newInf, h_newInf, NP * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cellIdx, h_cellIdx, NP * sizeof(int), cudaMemcpyHostToDevice);

    free(h_x);
    free(h_y);
    free(h_incub);
    free(h_susc);
    free(h_newInf);
    // free(h_cellIdx);

    log_memory_usage("After host→device memcpy");

    int threads = 256;
    int blocks = (NP + threads - 1) / threads;

    rebuildCellMap(
        d_cellIdx,
        d_x,
        d_y,
        d_susc,
        d_incub,
        d_newInf,
        d_cellStart,
        d_cellCount);

    log_memory_usage("After rebuildCellMap");

    // show to me the situation of the device arrays after the rebuildCellMap
    // so recreate the structures on the host and copy the data from the device to the host
    // then print the situation of the arrays in a way that is really readable
    int *h_cellStart = (int *)malloc(W * H * sizeof(int));
    int *h_cellCount = (int *)malloc(W * H * sizeof(int));
    cudaMemcpy(h_cellIdx, d_cellIdx, NP * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cellStart, d_cellStart, W * H * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cellCount, d_cellCount, W * H * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Cell index:\n");
    for (int i = 0; i < NP; ++i)
    {
        printf("%d ", h_cellIdx[i]);
    }
    printf("Cell start:\n");
    for (int i = 0; i < W * H; ++i)
    {
        printf("%d ", h_cellStart[i]);
    }
    printf("\nCell count:\n");
    for (int i = 0; i < W * H; ++i)
    {
        printf("%d ", h_cellCount[i]);
    }
    printf("\n");

    init_curand_kernel<<<blocks, threads>>>(d_curandStates, time(NULL));

    for (int day = 0; day < ND; ++day)
    {
        infect_kernel<<<blocks, threads>>>(
            d_x,
            d_y,
            d_incub,
            d_susc,
            d_newInf,
            d_cellStart,
            d_cellCount);

        status_kernel<<<blocks, threads>>>(
            d_x,
            d_y,
            d_incub,
            d_susc,
            d_newInf,
            d_curandStates);

        cudaDeviceSynchronize();

        rebuildCellMap(
            d_cellIdx,
            d_x,
            d_y,
            d_susc,
            d_incub,
            d_newInf,
            d_cellStart,
            d_cellCount);

        cudaMemcpy(h_cellIdx, d_cellIdx, NP * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_cellStart, d_cellStart, W * H * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_cellCount, d_cellCount, W * H * sizeof(int), cudaMemcpyDeviceToHost);
        printf("Cell index:\n");
        for (int i = 0; i < NP; ++i)
        {
            printf("%d ", h_cellIdx[i]);
        }
        printf("Cell start:\n");
        for (int i = 0; i < W * H; ++i)
        {
            printf("%d ", h_cellStart[i]);
        }
        printf("\nCell count:\n");
        for (int i = 0; i < W * H; ++i)
        {
            printf("%d ", h_cellCount[i]);
        }
        printf("\n");
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