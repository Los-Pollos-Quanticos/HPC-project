#include "../utils.h"
#include "../tupleList.h"
#include "../occupancyMap.h"

Cell *occupancy_map = NULL;

void init_population(Person *population)
{
    // --- Check if population fits grid (NP <= W * H * MAXP_CELL) ---
    if (NP > W * H * MAXP_CELL)
    {
        printf("Error: Population size exceeds available space on the grid.\n");
        exit(1); // Exits the program if the population size exceeds capacity
    }

    int i;
    int num_immune = (int)(NP * IMM);
    int num_infected = (int)(NP * INFP);

    // --- Array with available coordinates and occupancy count ---
    occupancy_map = malloc(W * H * sizeof(Cell));

    if (occupancy_map == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for occupancy map.\n");
        return;
    }

    TList *available_coords = createTList(W * H);
    for (int x = 0; x < W; x++)
    {
        for (int y = 0; y < H; y++)
        {
            addTuple(available_coords, x, y);
            AT(x, y).occupancy = 0;
            AT(x, y).persons = malloc(MAXP_CELL * sizeof(Person *));
            if (AT(x, y).persons == NULL)
            {
                fprintf(stderr, "Failed to allocate persons array for cell %d\n", i);
                exit(1);
            }
        }
    }

    // --- Initialize persons ---
    for (i = 0; i < NP; i++)
    {
        Person *p = &population[i];
        Tuple t;
        int idx = getRandomTupleIndex(available_coords, &t);

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

void simulate_one_day(Person *population)
{
    for (int i = 0; i < NP; i++)
    {
        Person *p = &population[i];
        if (is_dead(p))
            continue;

        // spread infection
        if (is_infected(p))
        {
            p->incubation_days--;

            for (int dx = -1; dx <= 1; dx++)
            {
                for (int dy = -1; dy <= 1; dy++)
                {
                    int nx = p->x + dx;
                    int ny = p->y + dy;

                    if (nx < 0 || nx >= W || ny < 0 || ny >= H)
                        continue;

                    for (int j = 0; j < AT(nx, ny).occupancy; j++)
                    {
                        Person *neighbor = AT(nx, ny).persons[j];
                        if (!is_infected(neighbor) && !is_immune(neighbor))
                        {
                            float infectivity = BETA * neighbor->susceptibility;
                            if (infectivity > ITH)
                            {
                                neighbor->incubation_days = INCUBATION_DAYS + 1; // new infection
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
        bool occupancy_valid = AT(new_x, new_y).occupancy < MAXP_CELL;

        if (xy_valid && occupancy_valid)
        {
            // Update occupancy map
            movePerson(p, new_x, new_y);

            // Update position
            p->x = new_x;
            p->y = new_y;
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
}

int main()
{
    Person *population = (Person *)malloc(NP * sizeof(Person));

    if (population == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    srand(time(NULL));

    init_population(population);
    // stats(population);
    // print_occupancies_map(population);

    // simulation
    for (int day = 0; day < ND; day++)
    {
        simulate_one_day(population);
        print_daily_report(population, day);
        print_occupancies_map(population);
    }

    free(population);
    free(occupancy_map);
    occupancy_map = NULL;
}
