#include "utils_cuda.cuh"

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
        int idx = getRandomTupleIndexSerial(available_coords, &t);

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

void debugState(const char *phase,
                int *d_x, int *d_y,
                int *d_incub, float *d_susc,
                int *d_cellCount)
{
    const int P_SAMPLE = NP; // show all persons
    const int G_PATCH = W;   // full grid width in this tiny example

    printf("===== DEBUG: %s =====\n", phase);

    // 1) copy down all person data
    int h_x[P_SAMPLE], h_y[P_SAMPLE], h_incub[P_SAMPLE];
    float h_susc[P_SAMPLE];
    cudaMemcpy(h_x, d_x, sizeof(int) * P_SAMPLE, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y, d_y, sizeof(int) * P_SAMPLE, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_incub, d_incub, sizeof(int) * P_SAMPLE, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_susc, d_susc, sizeof(float) * P_SAMPLE, cudaMemcpyDeviceToHost);

    // 2) pretty‐print a table of each person
    printf(" Persons:\n");
    printf("  ID |   (x,y)   | IncubDays | Susc\n");
    printf(" ----+-----------+-----------+------\n");
    for (int i = 0; i < P_SAMPLE; ++i)
    {
        printf(" %3d | (%2d,%2d) |     %2d    | %.2f\n",
               i,
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

void save_population(
    const int *d_x,
    const int *d_y,
    const int *d_incub,
    const float *d_susc,
    int day)
{
    int *h_x = (int *)malloc(NP * sizeof(int));
    int *h_y = (int *)malloc(NP * sizeof(int));
    int *h_incub = (int *)malloc(NP * sizeof(int));
    float *h_susc = (float *)malloc(NP * sizeof(float));

    if (!h_x || !h_y || !h_incub || !h_susc)
    {
        fprintf(stderr, "utils_cuda: malloc failed in save_population_cuda\n");
        free(h_x);
        free(h_y);
        free(h_incub);
        free(h_susc);
    }

    cudaMemcpy(h_x, d_x, NP * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y, d_y, NP * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_incub, d_incub, NP * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_susc, d_susc, NP * sizeof(float), cudaMemcpyDeviceToHost);

    char filename[64];
    snprintf(filename, sizeof(filename), "./report/day_%03d.dat", day);
    FILE *f = fopen(filename, "wb");
    if (!f)
    {
        fprintf(stderr, "utils_cuda: failed to open %s for writing\n", filename);
        free(h_x);
        free(h_y);
        free(h_incub);
        free(h_susc);
    }

    int np_val = NP;
    fwrite(&np_val, sizeof(int), 1, f);

    for (int i = 0; i < NP; ++i)
    {
        int state;
        if (h_x[i] < 0 || h_y[i] < 0)
            state = DEAD;
        else if (h_susc[i] == 0.0f)
            state = IMMUNE;
        else if (h_incub[i] > 0)
            state = INFECTED;
        else
            state = SUSCEPTIBLE;

        PersonReport pr;
        pr.x = h_x[i];
        pr.y = h_y[i];
        pr.state = (State)state;
        fwrite(&pr, sizeof(PersonReport), 1, f);
    }
    fclose(f);

    free(h_x);
    free(h_y);
    free(h_incub);
    free(h_susc);
}