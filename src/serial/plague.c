#include "../utils/utils.h"
#include "../structures/tupleList.h"
#include "../structures/occupancyMap.h"

#include <time.h>

Cell *occupancy_map = NULL;
Person **all_persons_pointers = NULL;

void init_population(Person *population)
{
    int grid_size = W * H;
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

    all_persons_pointers = (Person **)malloc((long)W * H * MAXP_CELL * sizeof(Person *));
    if (all_persons_pointers == NULL)
    {
        fprintf(stderr, "Failed to allocate global persons pointers array.\n");
        exit(1);
    }

    TList *available_coords = createTList(grid_size);

    for (int x = 0; x < W; x++)
    {
        for (int y = 0; y < H; y++)
        {
            addTuple(available_coords, x, y);
            AT(x, y).occupancy = 0;
            AT(x, y).persons = &all_persons_pointers[(x * H + y) * MAXP_CELL];
            for (int k = 0; k < MAXP_CELL; k++)
            {
                AT(x, y).persons[k] = NULL;
            }
        }
    }

    for (i = 0; i < NP; i++)
    {
        unsigned int seed = (unsigned int)rand();
        Person *p = &population[i];
        Tuple t;
        int idx = getRandomTupleIndexSerial(available_coords, &t);

        p->x = t.x;
        p->y = t.y;
        addPerson(p, t.x, t.y);

        if (i < num_immune)
        {
            p->susceptibility = 0.0f;
            p->incubation_days = 0;
        }
        else
        {
            p->incubation_days = (i < num_immune + num_infected) ? INCUBATION_DAYS + 1 : 0;
            p->susceptibility = gaussian_random(seed, S_AVG, 0.1f);
        }

        p->new_infected = false;

        if (AT(t.x, t.y).occupancy == MAXP_CELL)
        {
            removeTupleAt(available_coords, idx);
        }
    }

    freeTList(available_coords);
}

void simulate_one_day(Person *population)
{
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
                p->incubation_days = 0;
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
}

int main(int argc, char **argv)
{
    bool debug = false;
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--debug"))
            debug = true;

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

    // simulation
    for (int day = 0; day < ND; day++)
    {
        if (debug)
        {
            save_population(population, day);
        }
        simulate_one_day(population);
    }

    free(population);
    free(occupancy_map);
    occupancy_map = NULL;
    free(all_persons_pointers);
    all_persons_pointers = NULL;

    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("RunTime: %ld ms\n", get_time_in_ms(start, end));
}
