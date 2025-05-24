#include "../utils/utils.h"
#include "../structures/tupleList.h"
#include "../structures/occupancyMap.h"

#include <time.h>

Cell *occupancy_map = NULL;

void init_population(Person *population)
{

    clock_t start_occupancy_map_time = clock();
    // --- Check if population fits grid (NP <= W * H * MAXP_CELL) ---
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

    clock_t end_occupancy_map_time = clock();
    double occupancy_map_time = (double)(end_occupancy_map_time - start_occupancy_map_time) / CLOCKS_PER_SEC;
    //printf("Time taken to initialize occupancy map: %f seconds\n", occupancy_map_time);


    clock_t start_population_time = clock();
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

    freeTList(available_coords);
    clock_t end_population_time = clock();
    double population_time = (double)(end_population_time - start_population_time) / CLOCKS_PER_SEC;
    //printf("Time taken to distribute population: %f seconds\n", population_time);
}

void simulate_one_day(Person *population)
{
    clock_t start_simulation_time = clock();
    int max_num_new_infected = NP - (int)(NP * IMM);
    Person **newly_infected = malloc(sizeof(Person *) * max_num_new_infected);
    int newly_count = 0;

    for (int i = 0; i < NP; i++)
    {
        Person *p = &population[i];
        if (is_dead(p))
            continue;

        // spread infection
        if (is_infected(p))
        {
            p->incubation_days--;

            for (int dx = -IRD; dx <= IRD; dx++)
            {
                for (int dy = -IRD; dy <= IRD; dy++)
                {
                    int nx = p->x + dx;
                    int ny = p->y + dy;

                    if (nx < 0 || nx >= W || ny < 0 || ny >= H)
                        continue;

                    Cell *cell = &AT(nx, ny);
                    for (int j = 0; j < cell->occupancy; j++)
                    {
                        Person *neighbor = cell->persons[j];
                        if (!is_infected(neighbor) && !is_immune(neighbor) && !neighbor->new_infected)
                        {
                            float infectivity = BETA * neighbor->susceptibility;
                            if (infectivity > ITH)
                            {
                                neighbor->new_infected = true;
                                newly_infected[newly_count++] = neighbor;
                            }
                        }
                    }
                }
            }
        }

        // move randomly in an adjacent cell or stay still
        int dx = (rand() % 3) - 1; // -1, 0, or 1
        int dy = (rand() % 3) - 1; // -1, 0, or 1

        int new_x = p->x + dx;
        int new_y = p->y + dy;

        bool xy_valid = new_x >= 0 && new_x < W && new_y >= 0 && new_y < H;

        if (xy_valid)
        {
            bool occupancy_valid = AT(new_x, new_y).occupancy < MAXP_CELL;
            if (occupancy_valid)
            {
                // Update occupancy map
                movePerson(p, new_x, new_y);

                // Update position
                p->x = new_x;
                p->y = new_y;
            }
        }

        // Check last day of incubation
        if (p->incubation_days == 1)
        {
            float prob = (float)rand() / RAND_MAX;
            if (prob < MU)
            {
                // recover
                p->incubation_days = 0;

                if (rand() % 2 == 0)
                {
                    // become immune
                    p->susceptibility = 0.0f;
                }
            }
            else
            {
                removePerson(p);
                // die
                p->x = -1;
                p->y = -1;
            }
        }
    }

    for (int i = 0; i < newly_count; i++)
    {
        Person *p = newly_infected[i];
        p->new_infected = false;
        p->incubation_days = INCUBATION_DAYS + 1;
    }
    free(newly_infected);
    clock_t end_simulation_time = clock();
    double simulation_time = (double)(end_simulation_time - start_simulation_time) / CLOCKS_PER_SEC;
    //printf("Time taken to simulate day: %f seconds\n", simulation_time);
}

int main()
{
    //printf("Simulation of a plague started\n");
    Person *population = (Person *)malloc(NP * sizeof(Person));

    if (population == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    srand(time(NULL));

    
    clock_t start_time = clock();
    init_population(population);
    clock_t init_pop_time = clock();

    clock_t simulate_day_time = clock();
    // simulation
    for (int day = 0; day < ND; day++)
    {
        printf("------- DAY %d DEBUG --------\n", day);
        save_population(population, day);
        simulate_one_day(population);
    }

    clock_t end_time = clock();

    
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    double simulate_day_time_elapsed = (double)(end_time - simulate_day_time) / CLOCKS_PER_SEC;
    double init_pop_time_elapsed = (double)(init_pop_time - start_time) / CLOCKS_PER_SEC;
    printf("Simulation completed in %f seconds\n", elapsed_time);
    printf("Days simulated in %f seconds\n", simulate_day_time_elapsed);
    printf("Population initialized in %f seconds\n", init_pop_time_elapsed);
    free(population);
    freeOccupancyMap();
    occupancy_map = NULL;
}
