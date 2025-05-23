#include "../utils/utils.h"
#include "../structures/occupancyMap.h"

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
    int x;
    int y;
} Tuple;

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

    // create an array of coords that is large a grid_capacity then make a shuffle
    // to make the first NP coords random

    Tuple *coords = malloc(grid_capacity * sizeof(Tuple));

    if (coords == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for coords array.\n");
        return;
    }

    // fill coords in a sequential manner so placing the first x and y for MAXP_CELL times and so on
    int idx = 0;
    for (int x = 0; x < W; x++)
    {
        for (int y = 0; y < H; y++)
        {
            AT(x, y).occupancy = 0;
            AT(x, y).persons = malloc(MAXP_CELL * sizeof(Person *));
            if (AT(x, y).persons == NULL)
            {
                fprintf(stderr, "Failed to allocate persons array for cell %d\n", x * H + y);
                exit(1);
            }
            // Fill coords array with MAXP_CELL copies of each (x, y)
            for (int k = 0; k < MAXP_CELL; k++)
            {
                if (idx < grid_capacity)
                {
                    coords[idx].x = x;
                    coords[idx].y = y;
                    idx++;
                }
            }
        }
    }

    // Shuffle the coords array to randomize the order shuffle just the first NP elements
    unsigned int seed = (unsigned int)time(NULL);
    for (int i = 0; i < NP; i++)
    {
        int range = grid_capacity - i;
        int j = (range == 0) ? i : i + rand_r(&seed) % range;
        Tuple temp = coords[i];
        coords[i] = coords[j];
        coords[j] = temp;
    }

    // --- Initialize persons ---
    for (i = 0; i < NP; i++)
    {
        Person *p = &population[i];
        Tuple t = coords[i];

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
    }
    printf("Memory used before TLfree: %.2f MB\n", get_memory_usage_mb());
    free(coords);
    coords = NULL;
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
