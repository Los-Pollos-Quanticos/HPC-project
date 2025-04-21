#include <omp.h>

#include "../utils.h"
#include "../tupleList.h"
#include "../occupancyMap.h"

#define NTHREADS 12

Cell *occupancy_map = NULL;

void init_population(Person *population)
{
    if (NP > W * H * MAXP_CELL)
    {
        printf("Error: Population size exceeds available space on the grid.\n");
        exit(1); 
    }

    int num_immune = (int)(NP * IMM);
    int num_infected = (int)(NP * INFP);

    occupancy_map = malloc(W * H * sizeof(Cell));

    if (occupancy_map == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for occupancy map.\n");
        return;
    }

    TList *available_coords = createTList(W * H);
    int local_dim = (W * H) / NTHREADS + 1;

    #pragma omp parallel
    {
        TList* local_coords = createTList(local_dim);

        #pragma omp for collapse(2) nowait
        for (int x = 0; x < W; x++)
        {
            for (int y = 0; y < H; y++)
            {
                addTuple(local_coords, x, y);
                AT(x, y).occupancy = 0;
                AT(x, y).persons = malloc(MAXP_CELL * sizeof(Person *));
                if (AT(x, y).persons == NULL)
                {
                    fprintf(stderr, "Failed to allocate persons array for cell (%d,%d)\n", x, y);
                    exit(1);
                }
            }
        }

        //Each thread will now separately do:
        #pragma omp critical
        {
            for (int i = 0; i < local_coords->size; i++)
            {
                addTuple(available_coords, local_coords->data[i].x, local_coords->data[i].y);
            }
        }

        freeTList(local_coords);
    }
  

    // --- Initialize persons ---
    #pragma omp parallel for
    for (int i = 0; i < NP; i++)
    {
        Person *p = &population[i];
        Tuple t;

        int idx;
        #pragma omp critical
        {
            idx = getRandomTupleIndex(available_coords, &t);
        }

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
            p->susceptibility = gaussian_random(S_AVG, 0.1f);
            p->incubation_days = INCUBATION_DAYS + 1;
            p->new_infected = false;
        }
        else // Susceptible
        {
            p->susceptibility = gaussian_random(S_AVG, 0.1f);
            p->incubation_days = 0;
        }

        if (AT(t.x, t.y).occupancy == MAXP_CELL)
        {
            removeTupleAt(available_coords, idx);
        }
    }

    freeTList(available_coords);
}


/*rand() is not guaranteed to be thread-safe.
POSIX offered a thread-safe version of rand called rand_r, 
which is obsolete in favor of the drand48 family of functions.
This version uses with rand_r since VSCode on Windows uses doesn’t 
find the full definition of "struct drand48_data" */

void simulate_one_day(Person *population)
{
    int max_num_new_infected = NP - (int)(NP * IMM);
    Person **newly_infected = malloc(sizeof(Person *) * max_num_new_infected);
    int newly_count = 0;

    omp_lock_t cell_locks[W][H];
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < W; i++)
        for (int j = 0; j < H; j++)
            omp_init_lock(&cell_locks[i][j]);

    #pragma omp parallel
    {
        unsigned int seed = (unsigned int)(time(NULL) ^ omp_get_thread_num());
        Person **local_newly_infected = malloc(sizeof(Person *) * max_num_new_infected);
        int local_newly_count = 0;

        #pragma omp for schedule(guided)
        for (int i = 0; i < NP; i++) {
            Person *p = &population[i];

            if (is_dead(p)) continue;

            if (is_infected(p)) {
                p->incubation_days--;

                for (int dx = -IRD; dx <= IRD; dx++) {
                    for (int dy = -IRD; dy <= IRD; dy++) {
                        int nx = p->x + dx;
                        int ny = p->y + dy;

                        if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;

                        omp_set_lock(&cell_locks[nx][ny]);

                        for (int j = 0; j < AT(nx, ny).occupancy; j++) {
                            Person *neighbor = AT(nx, ny).persons[j];
                            if (!is_infected(neighbor) && !is_immune(neighbor)) {
                                float infectivity = BETA * neighbor->susceptibility;
                                if (infectivity > ITH) {
                                    neighbor->new_infected = true;
                                    local_newly_infected[local_newly_count++] = neighbor;
                                }
                            }
                        }

                        omp_unset_lock(&cell_locks[nx][ny]);
                    }
                }
            }

            int dx = (rand_r(&seed) % 3) - 1;
            int dy = (rand_r(&seed) % 3) - 1;
            int new_x = p->x + dx;
            int new_y = p->y + dy;

            bool xy_valid = new_x >= 0 && new_x < W && new_y >= 0 && new_y < H;

            if (xy_valid) {
                omp_set_lock(&cell_locks[new_x][new_y]);
                bool occupancy_valid = AT(new_x, new_y).occupancy < MAXP_CELL;

                if (occupancy_valid) {
                    movePerson(p, new_x, new_y);
                    p->x = new_x;
                    p->y = new_y;
                }

                omp_unset_lock(&cell_locks[new_x][new_y]);
            }

            // Infection resolution
            if (p->incubation_days == 1) {
                float prob = (float)rand_r(&seed) / RAND_MAX;

                if (prob < MU) {
                    p->incubation_days = 0;

                    if (rand_r(&seed) % 2 == 0)
                        p->susceptibility = 0.0f;

                } else {
                    omp_set_lock(&cell_locks[p->x][p->y]);

                    removePerson(p);
                    omp_unset_lock(&cell_locks[p->x][p->y]);
                    p->x = -1;
                    p->y = -1;
                }
            }
        }

        #pragma omp critical
        {
            for (int i = 0; i < local_newly_count; i++)
                newly_infected[newly_count++] = local_newly_infected[i];
        }

        free(local_newly_infected);
    }

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < newly_count; i++) {
            Person *p = newly_infected[i];
            p->new_infected = false;
            p->incubation_days = INCUBATION_DAYS + 1;
        }
    
        #pragma omp for collapse(2)
        for (int i = 0; i < W; i++)
            for (int j = 0; j < H; j++)
                omp_destroy_lock(&cell_locks[i][j]);
        
    }   

    free(newly_infected);
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
        simulate_one_day(population);
    }

    free(population);
    free(occupancy_map);
    occupancy_map = NULL;
}
