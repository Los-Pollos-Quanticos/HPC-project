#include <omp.h>

#include "../utils/utils.h"
#include "../structures/tupleList.h"
#include "../structures/occupancyMap.h"

#define NTHREADS 6
#define TOT_CELL (W * H)
#define LOCK(x, y) cell_locks[x * H + y]

Cell *occupancy_map = NULL;
Person **all_persons_pointers = NULL;
omp_lock_t *cell_locks = NULL;
bool debug = false;

/**
 * @brief Initializes the OpenMP locks for each cell in the grid.
 * @param None
 * @return None
 */
void init_locks()
{
    cell_locks = malloc((long)TOT_CELL * sizeof(omp_lock_t));
    if (cell_locks == NULL)
    {
        fprintf(stderr, "Failed to allocate cell locks.\n");
        exit(1);
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < W; i++)
        for (int j = 0; j < H; j++)
            omp_init_lock(&LOCK(i, j));
}

/**
 * @brief Destroys the OpenMP locks and frees the allocated memory.
 * @param None
 * @return None
 */
void destroy_locks()
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < W; i++)
        for (int j = 0; j < H; j++)
            omp_destroy_lock(&LOCK(i, j));

    free(cell_locks);
}

/**
 * @brief Initializes the simulation environment and distributes the population.
 * @param population A pointer to an array of `Person` structures representing the entire population.
 * @return None
 */
void init_population(Person *population)
{
    if (debug)
        printf("------ STARTING INIT POPULATION ------\n");

    if (NP > W * H * MAXP_CELL)
    {
        printf("Error: Population size exceeds available space on the grid.\n");
        exit(1);
    }

    occupancy_map = malloc(W * H * sizeof(Cell));
    if (occupancy_map == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for occupancy map.\n");
        exit(1);
    }

    all_persons_pointers = (Person **)malloc((long)W * H * MAXP_CELL * sizeof(Person *));
    if (all_persons_pointers == NULL)
    {
        fprintf(stderr, "Failed to allocate global persons pointers array.\n");
        exit(1);
    }

    int num_immune = (int)(NP * IMM);
    int num_infected = (int)(NP * INFP);

#pragma omp parallel for collapse(2)
    for (int x = 0; x < W; x++)
    {
        for (int y = 0; y < H; y++)
        {
            AT(x, y).occupancy = 0;
            AT(x, y).persons = &all_persons_pointers[(x * H + y) * MAXP_CELL];
            for (int k = 0; k < MAXP_CELL; k++)
            {
                AT(x, y).persons[k] = NULL;
            }
        }
    }

    if (debug)
    {
        printf("Number of cells: %d, Number of persons: %d, Number of threads: %d\n", W * H, NP, NTHREADS);
        printf("Number of immune: %d, infected: %d\n", num_immune, num_infected);
    }

    // Each thread will have its available coordinates and occupancy count
    int *cells_per_thread = malloc(NTHREADS * sizeof(int));
    int *people_per_thread = malloc(NTHREADS * sizeof(int));
    int *cell_offset = malloc(NTHREADS * sizeof(int));
    int *people_offset = malloc(NTHREADS * sizeof(int));
    int *immune_per_thread = malloc(NTHREADS * sizeof(int));
    int *infected_per_thread = malloc(NTHREADS * sizeof(int));

    // Calculation of the number of cells each thread will handle
    for (int i = 0; i < NTHREADS; ++i)
        cells_per_thread[i] = (TOT_CELL) / NTHREADS + (i < (TOT_CELL) % NTHREADS ? 1 : 0);

    int assigned_people = 0;
    int assigned_immune = 0;
    int assigned_infected = 0;

    // Susceptible, immune, and infected are distributed among
    // threads proportionally to the number of cells they handle
    for (int i = 0; i < NTHREADS; i++)
    {
        int max_people = MAXP_CELL * cells_per_thread[i];
        people_per_thread[i] = (int)(((long)NP * cells_per_thread[i]) / (TOT_CELL));
        assigned_people += people_per_thread[i];

        immune_per_thread[i] = (int)(((long)num_immune * cells_per_thread[i]) / (TOT_CELL));
        assigned_immune += immune_per_thread[i];

        infected_per_thread[i] = (int)(((long)num_infected * cells_per_thread[i]) / (TOT_CELL));
        assigned_infected += infected_per_thread[i];
    }

    int remaining_people = NP - assigned_people;
    int remaining_immune = num_immune - assigned_immune;
    int remaining_infected = num_infected - assigned_infected;

    // Remaining people, immune, and infected are distributed starting from the first thread
    for (int i = 0; i < NTHREADS && (remaining_people > 0 || remaining_immune > 0 || remaining_infected > 0); i++)
    {
        int max_people = MAXP_CELL * cells_per_thread[i];

        while (people_per_thread[i] < max_people && remaining_people > 0)
        {
            people_per_thread[i]++;
            remaining_people--;
        }

        while (immune_per_thread[i] < people_per_thread[i] && remaining_immune > 0)
        {
            immune_per_thread[i]++;
            remaining_immune--;
        }

        while (infected_per_thread[i] < (people_per_thread[i] - immune_per_thread[i]) && remaining_infected > 0)
        {
            infected_per_thread[i]++;
            remaining_infected--;
        }
    }

    if (debug)
    {
        printf("\nFinal cells_per_thread: ");
        for (int i = 0; i < NTHREADS; ++i)
            printf("%d ", cells_per_thread[i]);
        printf("\nFinal people_per_thread: ");
        for (int i = 0; i < NTHREADS; ++i)
            printf("%d ", people_per_thread[i]);
        printf("\nFinal immune_per_thread: ");
        for (int i = 0; i < NTHREADS; ++i)
            printf("%d ", immune_per_thread[i]);
        printf("\nFinal infected_per_thread: ");
        for (int i = 0; i < NTHREADS; ++i)
            printf("%d ", infected_per_thread[i]);
        printf("\n");
    }

    // Each thread will work on a specific range of cells and people, given by these offsets.
    // The first thread starts from person and cell "zero", the others are calculated based on the previous thread's end
    cell_offset[0] = people_offset[0] = 0;
    for (int i = 1; i < NTHREADS; i++)
    {
        cell_offset[i] = cell_offset[i - 1] + cells_per_thread[i - 1];
        people_offset[i] = people_offset[i - 1] + people_per_thread[i - 1];
    }

    if (debug)
    {
        printf("\nFinal cell_offset: ");
        for (int i = 0; i < NTHREADS; ++i)
            printf("%d ", cell_offset[i]);
        printf("\nFinal people_offset: ");
        for (int i = 0; i < NTHREADS; ++i)
            printf("%d ", people_offset[i]);
        printf("\n\n");
    }

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        unsigned int seed = (unsigned int)(time(NULL) ^ tid);

        int cell_start = cell_offset[tid];
        int cell_end = cell_start + cells_per_thread[tid];

        int people_start = people_offset[tid];
        int people_end = people_start + people_per_thread[tid];

        TList *local_coords = createTList(cells_per_thread[tid]);

        // Tuple generations for each thread
        for (int i = cell_start; i < cell_end; i++)
        {
            int x = i / H;
            int y = i % H;
            addTuple(local_coords, x, y);
        }

        if (debug && TOT_CELL <= 100)
        {
#pragma omp critical
            {
                printf("Thread %d local coordinates: ", tid);
                for (int i = 0; i < local_coords->size; i++)
                {
                    printf("(%d, %d) ", local_coords->data[i].x, local_coords->data[i].y);
                }
                printf("\n");
            }

#pragma omp barrier
        }

        int person_count = 0;

        // Population distribution and initialization
        for (int i = people_start; i < people_end; i++)
        {
            Person *p = &population[i];
            Tuple t;

            int idx = getRandomTupleIndex(seed, local_coords, &t);

            p->x = t.x;
            p->y = t.y;

            addPerson(p, t.x, t.y);

            int role = 2;
            if (person_count < immune_per_thread[tid])
                role = 0;
            else if (person_count < immune_per_thread[tid] + infected_per_thread[tid])
                role = 1;

            if (role == 0)
            {
                p->susceptibility = 0.0f;
                p->incubation_days = 0;
            }
            else if (role == 1)
            {
                p->susceptibility = gaussian_random(seed, S_AVG, 0.1f);
                p->incubation_days = INCUBATION_DAYS + 1;
                p->new_infected = false;
            }
            else
            {
                p->susceptibility = gaussian_random(seed, S_AVG, 0.1f);
                p->incubation_days = 0;
            }

            if (AT(t.x, t.y).occupancy == MAXP_CELL)
                removeTupleAt(local_coords, idx);

            person_count++;
        }

        freeTList(local_coords);
    }

    free(cells_per_thread);
    free(people_per_thread);
    free(cell_offset);
    free(people_offset);
    free(immune_per_thread);
    free(infected_per_thread);
}

/**
 * @brief Simulates one day of activity for the population.
 * @param population A pointer to an array of `Person` structures representing the entire population.
 * @return None
 */
void simulate_one_day(Person *population)
{
    int max_num_new_infected = NP - (int)(NP * IMM);

#pragma omp parallel
    {
        unsigned int seed = (unsigned int)(time(NULL) ^ omp_get_thread_num());
        Person **local_newly_infected = malloc(sizeof(Person *) * max_num_new_infected);
        int local_newly_count = 0;

#pragma omp for schedule(guided)
        for (int i = 0; i < NP; i++)
        {
            Person *p = &population[i];

            if (is_dead(p))
                continue;

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

                        // Acquire a lock for the neighbor's cell to prevent race conditions during access.
                        omp_set_lock(&LOCK(nx, ny));

                        // For each neighbo, check if it is susceptible
                        for (int j = 0; j < AT(nx, ny).occupancy; j++)
                        {
                            Person *neighbor = AT(nx, ny).persons[j];
                            if (!is_infected(neighbor) && !is_immune(neighbor) && !neighbor->new_infected)
                            {
                                float infectivity = BETA * neighbor->susceptibility;
                                if (infectivity > ITH)
                                {
                                    neighbor->new_infected = true;
                                    // Newly infected persons are stored in a local array for later processing
                                    local_newly_infected[local_newly_count++] = neighbor;
                                }
                            }
                        }

                        omp_unset_lock(&LOCK(nx, ny));
                    }
                }
            }

            int dx = (rand_r(&seed) % 3) - 1;
            int dy = (rand_r(&seed) % 3) - 1;
            int new_x = p->x + dx;
            int new_y = p->y + dy;

            bool xy_valid = new_x >= 0 && new_x < W && new_y >= 0 && new_y < H &&
                            (new_x != p->x || new_y != p->y);

            if (xy_valid)
            {
                int x1 = p->x, y1 = p->y;
                int x2 = new_x, y2 = new_y;

                // The locks are acquired and left in a consistent order to prevent deadlocks.
                bool lock_current_first = (x1 < x2) || (x1 == x2 && y1 < y2);

                if (lock_current_first)
                {
                    omp_set_lock(&LOCK(x1, y1));
                    omp_set_lock(&LOCK(x2, y2));
                }
                else
                {
                    omp_set_lock(&LOCK(x2, y2));
                    omp_set_lock(&LOCK(x1, y1));
                }

                bool occupancy_valid = AT(new_x, new_y).occupancy < MAXP_CELL;

                if (occupancy_valid)
                {
                    movePerson(p, new_x, new_y);
                    p->x = new_x;
                    p->y = new_y;
                }

                if (lock_current_first)
                {
                    omp_unset_lock(&LOCK(x2, y2));
                    omp_unset_lock(&LOCK(x1, y1));
                }
                else
                {
                    omp_unset_lock(&LOCK(x1, y1));
                    omp_unset_lock(&LOCK(x2, y2));
                }
            }

            if (p->incubation_days == 1)
            {
                float prob = (float)rand_r(&seed) / RAND_MAX;

                if (prob < MU)
                {
                    p->incubation_days = 0;

                    if (rand_r(&seed) % 2 == 0)
                        p->susceptibility = 0.0f;
                }
                else
                {
                    omp_set_lock(&LOCK(p->x, p->y));
                    removePerson(p);
                    omp_unset_lock(&LOCK(p->x, p->y));
                    p->x = -1;
                    p->y = -1;
                }
            }
        }

        // Process the newly infected persons, which can infect others on the next day.
        for (int i = 0; i < local_newly_count; i++)
        {
            Person *p = local_newly_infected[i];
            p->new_infected = false;
            p->incubation_days = INCUBATION_DAYS + 1;
        }

        free(local_newly_infected);
    }
}

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--debug"))
            debug = true;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    omp_set_num_threads(NTHREADS);

    Person *population = (Person *)malloc(NP * sizeof(Person));

    if (population == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    srand(time(NULL));

    init_population(population);

    init_locks();

    for (int day = 0; day < ND; day++)
    {
        if (debug)
        {
            save_population(population, day);
            printf("------- DAY %d DEBUG --------\n", day);
        }
        simulate_one_day(population);
    }

    destroy_locks();

    free(population);
    free(occupancy_map);
    occupancy_map = NULL;
    free(all_persons_pointers);
    all_persons_pointers = NULL;

    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("RunTime: %ld ms\n", get_time_in_ms(start, end));
}
