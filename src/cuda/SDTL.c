#include "../utils/utils.h"
#include "../structures/tupleList.h"
#include "../structures/occupancyMap.h"

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

Cell *occupancy_map = NULL;

void init_population(Person *population)
{
    int grid_size = (int)W * H;
    int grid_capacity = (int)grid_size * MAXP_CELL;

    if (NP > grid_capacity)
    {
        printf("Error: Population size exceeds available space on the grid.\n");
        exit(1); // Exits the program if the population size exceeds capacity
    }

    int i;
    int num_immune = (int)(NP * IMM);
    int num_infected = (int)(NP * INFP);

    // --- Array with available coordinates and occupancy count ---
    occupancy_map = malloc(grid_size * sizeof(Cell));

    if (occupancy_map == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for occupancy map.\n");
        return;
    }

    TList *available_coords = createTList(grid_size);
    for (int x = 0; x < W; x++)
    {
        for (int y = 0; y < H; y++)
        {
            addTuple(available_coords, x, y);
            AT(x, y).occupancy = 0;
            AT(x, y).persons = malloc(MAXP_CELL * sizeof(Person *));
            if (AT(x, y).persons == NULL)
            {
                fprintf(stderr, "Failed to allocate persons array for cell %d\n", x * H + y);
                exit(1);
            }
        }
    }

    // --- Initialize persons ---
    for (i = 0; i < NP; i++)
    {
        Person *p = &population[i];
        Tuple t;
        unsigned int seed = (unsigned int)time(NULL);
        int idx = getRandomTupleIndex(seed, available_coords, &t);

        p->x = t.x;
        p->y = t.y;
        addPerson(p, t.x, t.y);

        int role = 2; // Default to susceptible
        if (i < num_immune)
            role = 0; // Immune
        else if (i < num_immune + num_infected)
            role = 1; // Infected

        if (role == 0) // Immune
        {
            p->susceptibility = 0.0f;
            p->incubation_days = 0;
        }
        else if (role == 1) // Infected
        {
            p->susceptibility = gaussian_random(seed, S_AVG, 0.1f);
            p->incubation_days = INCUBATION_DAYS + 1;
        }
        else // Susceptible
        {
            p->susceptibility = gaussian_random(seed, S_AVG, 0.1f);
            p->incubation_days = 0;
        }

        p->new_infected = false;

        if (AT(t.x, t.y).occupancy == MAXP_CELL)
        {
            removeTupleAt(available_coords, idx);
        }
    }
    printf("Memory used before TLfree: %.2f MB\n", get_memory_usage_mb());
    freeTList(available_coords);
}

int main()
{
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    Person *population = (Person *)malloc(NP * sizeof(Person));

    if (population == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    srand(time(NULL));

    init_population(population);

    printf("Memory used before free: %.2f MB\n", get_memory_usage_mb());
    free(population);
    freeOccupancyMap();
    occupancy_map = NULL;

    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Time: %ld ms\n", get_time_in_ms(start, end));
    printf("Memory used after free: %.2f MB\n", get_memory_usage_mb());
}
