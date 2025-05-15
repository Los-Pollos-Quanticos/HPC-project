#include <omp.h>

#include "../utils/utils.h"
#include "../structures/tupleList.h"
#include "../structures/occupancyMap.h"

#define NTHREADS 4

Cell *occupancy_map = NULL;

/*rand() is not guaranteed to be thread-safe.
POSIX offered a thread-safe version of rand called rand_r,
which is obsolete in favor of the drand48 family of functions.
This version uses rand_r.*/

void init_population(Person *population)
{

    printf("------ INIT POPULATION DEBUG ------\n");
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

    int num_immune = (int)(NP * IMM);
    int num_infected = (int)(NP * INFP);

    // print #cells #persons and #threads too
    printf("Number of cells: %d, Number of persons: %d, Number of threads: %d\n", W * H, NP, NTHREADS);
    printf("Number of immune: %d, infected: %d\n", num_immune, num_infected);

    #pragma omp parallel for collapse(2)
    for (int x = 0; x < W; x++)
    {
        for (int y = 0; y < H; y++)
        {
            AT(x, y).occupancy = 0;
            AT(x, y).persons = malloc(MAXP_CELL * sizeof(Person *));
            if (AT(x, y).persons == NULL)
            {
                fprintf(stderr, "Failed to allocate persons array at (%d, %d)\n", x, y);
                exit(1);
            }
        }
    }

    int total_cells = W * H;

    int *cells_per_thread = malloc(NTHREADS * sizeof(int));
    int *people_per_thread = malloc(NTHREADS * sizeof(int));
    int *cell_offset = malloc(NTHREADS * sizeof(int));
    int *people_offset = malloc(NTHREADS * sizeof(int));
    int *immune_per_thread = malloc(NTHREADS * sizeof(int));
    int *infected_per_thread = malloc(NTHREADS * sizeof(int));

    for (int i = 0; i < NTHREADS; ++i)
        cells_per_thread[i] = total_cells / NTHREADS + (i < total_cells % NTHREADS ? 1 : 0);

    // Videocall calculations
    int assigned_people = 0;
    int assigned_immune = 0;
    int assigned_infected = 0;

    for (int i = 0; i < NTHREADS; ++i)
    {
        // Distribute people
        int max_people = MAXP_CELL * cells_per_thread[i];
        people_per_thread[i] = (int)(((long)NP * cells_per_thread[i]) / total_cells);


        if (people_per_thread[i] > max_people) // Can it really overcome the max capacity?
            people_per_thread[i] = max_people;
        assigned_people += people_per_thread[i];

        // Distribute immune and infected
        immune_per_thread[i] = (int)(((long)num_immune * cells_per_thread[i]) / total_cells);
        infected_per_thread[i] = (int)(((long)num_infected * cells_per_thread[i]) / total_cells);

        if (immune_per_thread[i] > max_people)
            immune_per_thread[i] = max_people;
        assigned_immune += immune_per_thread[i];

        if (infected_per_thread[i] > max_people)
            infected_per_thread[i] = max_people;
        assigned_infected += infected_per_thread[i];
    }

    // We have now remaining people to assign, the immune and the infected too
    int remaining_people = NP - assigned_people;
    int remaining_immune = num_immune - assigned_immune;
    int remaining_infected = num_infected - assigned_infected;

    for (int i = 0; i < NTHREADS && (remaining_people > 0 || remaining_immune > 0 || remaining_infected > 0); i++)
    {
        int max_people = MAXP_CELL * cells_per_thread[i];

        while (people_per_thread[i] < max_people && remaining_people > 0)
        {
            people_per_thread[i]++;
            remaining_people--;
        }

        // Distribute remaining immune
        while (immune_per_thread[i] < people_per_thread[i] && remaining_immune > 0)
        {
            immune_per_thread[i]++;
            remaining_immune--;
        }

        // Distribute remaining infected
        while (infected_per_thread[i] < (people_per_thread[i] - immune_per_thread[i]) && remaining_infected > 0)
        {
            infected_per_thread[i]++;
            remaining_infected--;
        }
    }

    // print final arrays
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

    // To this point each thread knows how many persons to accomodate
    cell_offset[0] = people_offset[0] = 0;
    for (int i = 1; i < NTHREADS; i++)
    {
        cell_offset[i] = cell_offset[i - 1] + cells_per_thread[i - 1];
        people_offset[i] = people_offset[i - 1] + people_per_thread[i - 1];
    }

    // print the offsets arrays
    printf("\nFinal cell_offset: ");
    for (int i = 0; i < NTHREADS; ++i)
        printf("%d ", cell_offset[i]);
    printf("\nFinal people_offset: ");
    for (int i = 0; i < NTHREADS; ++i)
        printf("%d ", people_offset[i]);
    printf("\n\n");

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        unsigned int seed = (unsigned int)(time(NULL) ^ tid);

        int cell_start = cell_offset[tid];
        int cell_end = cell_start + cells_per_thread[tid];

        int people_start = people_offset[tid];
        int people_end = people_start + people_per_thread[tid];

        TList *local_coords = createTList(cells_per_thread[tid]);
        for (int i = cell_start; i < cell_end; i++)
        {
            int x = i / H;
            int y = i % H;
            addTuple(local_coords, x, y);
        }

        // each thread print its local coordinates
        // #pragma omp critical
        // {
        //     printf("Thread %d local coordinates: ", tid);
        //     for (int i = 0; i < local_coords->size; i++)
        //     {
        //         printf("(%d, %d) ", local_coords->data[i].x, local_coords->data[i].y);
        //     }
        //     printf("\n");
        // }

        // // DEBUG
        // #pragma omp barrier

        int person_count = 0;

        for (int i = people_start; i < people_end; i++)
        {
            Person *p = &population[i];
            Tuple t;

            int idx = getRandomTupleIndex(seed, local_coords, &t);

            p->x = t.x;
            p->y = t.y;

            addPerson(p, t.x, t.y);

            // Role assignment
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

            // printf("Thread %d adding person %d with days %d in (%d,%d)\n", tid, i, p->incubation_days, t.x, t.y);
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

// void simulate_one_day(Person *population)
// {
//     int max_num_new_infected = NP - (int)(NP * IMM);
//     // printf("Max number of infected: %d\n", max_num_new_infected);

//     omp_lock_t cell_locks[W][H];
//     #pragma omp parallel for collapse(2)
//     for (int i = 0; i < W; i++)
//         for (int j = 0; j < H; j++)
//             omp_init_lock(&cell_locks[i][j]);


//     #pragma omp parallel
//     {
//         unsigned int seed = (unsigned int)(time(NULL) ^ omp_get_thread_num());
//         Person **local_newly_infected = malloc(sizeof(Person *) * max_num_new_infected);
//         int local_newly_count = 0;

//         #pragma omp for schedule(guided)
//         for (int i = 0; i < NP; i++)
//         {
//             Person *p = &population[i];

//             // DEBUG
//             // printf("Thread %d taking person %d with incubation days %d in (%d,%d)\n", omp_get_thread_num(), i, p->incubation_days, p->x, p->y);

//             if (is_dead(p))
//                 continue;

//             if (is_infected(p))
//             {
//                 p->incubation_days--;

//                 for (int dx = -IRD; dx <= IRD; dx++)
//                 {
//                     for (int dy = -IRD; dy <= IRD; dy++)
//                     {
//                         int nx = p->x + dx;
//                         int ny = p->y + dy;

//                         if (nx < 0 || nx >= W || ny < 0 || ny >= H)
//                             continue;

//                         omp_set_lock(&cell_locks[nx][ny]);

//                         for (int j = 0; j < AT(nx, ny).occupancy; j++)
//                         {
//                             Person *neighbor = AT(nx, ny).persons[j];
//                             if (!is_infected(neighbor) && !is_immune(neighbor) && !neighbor->new_infected)
//                             {
//                                 float infectivity = BETA * neighbor->susceptibility;
//                                 if (infectivity > ITH)
//                                 {
//                                     neighbor->new_infected = true;
//                                     // printf("Person %d infected person %d in (%d,%d)\n", i, j, nx, ny);
//                                     local_newly_infected[local_newly_count++] = neighbor;
//                                 }
//                             }
//                         }

//                         omp_unset_lock(&cell_locks[nx][ny]);
//                     }
//                 }
//             }

//             int dx = (rand_r(&seed) % 3) - 1;
//             int dy = (rand_r(&seed) % 3) - 1;
//             int new_x = p->x + dx;
//             int new_y = p->y + dy;

//             bool xy_valid = new_x >= 0 && new_x < W && new_y >= 0 && new_y < H &&
//                             (new_x != p->x || new_y != p->y);

//             // if (xy_valid) {
//             //     omp_set_lock(&cell_locks[new_x][new_y]);
//             //     bool occupancy_valid = AT(new_x, new_y).occupancy < MAXP_CELL;

//             //     if (occupancy_valid) {
//             //         omp_set_lock(&cell_locks[p->x][p->y]);
//             //         movePerson(p, new_x, new_y);
//             //         omp_unset_lock(&cell_locks[p->x][p->y]);
//             //         printf("Thread %d moved person %d from (%d,%d) to (%d,%d)\n", omp_get_thread_num(), i, p->x, p->y, new_x, new_y);
//             //         p->x = new_x;
//             //         p->y = new_y;
//             //     }

//             //     omp_unset_lock(&cell_locks[new_x][new_y]);
//             // }

//             if (xy_valid)
//             {
//                 int x1 = p->x, y1 = p->y;
//                 int x2 = new_x, y2 = new_y;

//                 bool lock_current_first = (x1 < x2) || (x1 == x2 && y1 < y2);

//                 if (lock_current_first)
//                 {
//                     omp_set_lock(&cell_locks[x1][y1]);
//                     omp_set_lock(&cell_locks[x2][y2]);
//                 }
//                 else
//                 {
//                     omp_set_lock(&cell_locks[x2][y2]);
//                     omp_set_lock(&cell_locks[x1][y1]);
//                 }

//                 bool occupancy_valid = AT(new_x, new_y).occupancy < MAXP_CELL;

//                 if (occupancy_valid)
//                 {
//                     movePerson(p, new_x, new_y);
//                     // printf("Thread %d moved person %d from (%d,%d) to (%d,%d)\n",
//                     //        omp_get_thread_num(), i, x1, y1, new_x, new_y);
//                     p->x = new_x;
//                     p->y = new_y;
//                 }

//                 // Release locks in reverse order
//                 if (lock_current_first)
//                 {
//                     omp_unset_lock(&cell_locks[x2][y2]);
//                     omp_unset_lock(&cell_locks[x1][y1]);
//                 }
//                 else
//                 {
//                     omp_unset_lock(&cell_locks[x1][y1]);
//                     omp_unset_lock(&cell_locks[x2][y2]);
//                 }
//             }

//             // Infection resolution
//             if (p->incubation_days == 1)
//             {
//                 float prob = (float)rand_r(&seed) / RAND_MAX;

//                 if (prob < MU)
//                 {
//                     p->incubation_days = 0;

//                     if (rand_r(&seed) % 2 == 0)
//                         p->susceptibility = 0.0f;
//                 }
//                 else
//                 {
//                     omp_set_lock(&cell_locks[p->x][p->y]);
//                     removePerson(p);
//                     omp_unset_lock(&cell_locks[p->x][p->y]);
//                     p->x = -1;
//                     p->y = -1;
//                 }
//             }
//         }

//         #pragma omp for collapse(2)
//         for (int i = 0; i < W; i++)
//             for (int j = 0; j < H; j++)
//                 omp_destroy_lock(&cell_locks[i][j]);

//         for (int i = 0; i < local_newly_count; i++)
//         {
//             Person *p = local_newly_infected[i];
//             p->new_infected = false;
//             p->incubation_days = INCUBATION_DAYS + 1;
//         }

//         free(local_newly_infected);
//     }
// }

void simulate_one_day(Person *population)
{
    int max_num_new_infected = NP - (int)(NP * IMM);

    omp_lock_t *cell_locks = malloc(W * H * sizeof(omp_lock_t));
    if (cell_locks == NULL)
    {
        fprintf(stderr, "Failed to allocate cell locks.\n");
        exit(1);
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < W; i++)
        for (int j = 0; j < H; j++)
            omp_init_lock(&cell_locks[i * H + j]);

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

                        omp_set_lock(&cell_locks[nx * H + ny]);

                        for (int j = 0; j < AT(nx, ny).occupancy; j++)
                        {
                            Person *neighbor = AT(nx, ny).persons[j];
                            if (!is_infected(neighbor) && !is_immune(neighbor) && !neighbor->new_infected)
                            {
                                float infectivity = BETA * neighbor->susceptibility;
                                if (infectivity > ITH)
                                {
                                    neighbor->new_infected = true;
                                    local_newly_infected[local_newly_count++] = neighbor;
                                }
                            }
                        }

                        omp_unset_lock(&cell_locks[nx * H + ny]);
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

                bool lock_current_first = (x1 < x2) || (x1 == x2 && y1 < y2);

                if (lock_current_first)
                {
                    omp_set_lock(&cell_locks[x1 * H + y1]);
                    omp_set_lock(&cell_locks[x2 * H + y2]);
                }
                else
                {
                    omp_set_lock(&cell_locks[x2 * H + y2]);
                    omp_set_lock(&cell_locks[x1 * H + y1]);
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
                    omp_unset_lock(&cell_locks[x2 * H + y2]);
                    omp_unset_lock(&cell_locks[x1 * H + y1]);
                }
                else
                {
                    omp_unset_lock(&cell_locks[x1 * H + y1]);
                    omp_unset_lock(&cell_locks[x2 * H + y2]);
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
                    omp_set_lock(&cell_locks[p->x * H + p->y]);
                    removePerson(p);
                    omp_unset_lock(&cell_locks[p->x * H + p->y]);
                    p->x = -1;
                    p->y = -1;
                }
            }
        }

        #pragma omp for collapse(2)
        for (int i = 0; i < W; i++)
            for (int j = 0; j < H; j++)
                omp_destroy_lock(&cell_locks[i * H + j]);

        for (int i = 0; i < local_newly_count; i++)
        {
            Person *p = local_newly_infected[i];
            p->new_infected = false;
            p->incubation_days = INCUBATION_DAYS + 1;
        }

        free(local_newly_infected);
    }

    free(cell_locks);
}


int main()
{
    omp_set_num_threads(NTHREADS);

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
        save_population(population, day);
        printf("\n------- DAY %d DEBUG --------\n", day);
        simulate_one_day(population);
    }

    free(population);
    freeOccupancyMap();
    occupancy_map = NULL;
}
