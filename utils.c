#include "./utils.h"

// Gaussian random number through Box-Muller transform
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

void stats(Person *population)
{
    printf("Population details:\n");
    for (int i = 0; i < NP; i++)
    {
        printf("Person %d: Position (%d, %d), Susceptibility %.2f, Dead %d, Incubation Days %d\n",
               i, population[i].x, population[i].y, population[i].susceptibility,
               is_dead(&population[i]), population[i].incubation_days);
    }
}

bool is_dead(const Person *p)
{
    return p->x < 0 && p->y < 0;
}

bool is_immune(const Person *p)
{
    return p->susceptibility == 0.0f;
}

bool is_infected(const Person *p)
{
    return p->incubation_days > 0;
}

void print_daily_report(Person *population, int day)
{
    int dead = 0;
    int infected = 0;
    int immune = 0;
    int alive = 0;
    int new_infected = 0;
    float susceptibility_sum = 0.0f;
    int susceptible_count = 0;

    for (int i = 0; i < NP; i++)
    {
        Person *p = &population[i];

        if (is_dead(p))
        {
            dead++;
            continue;
        }

        alive++;

        if (is_infected(p))
        {
            infected++;

            if (p->incubation_days == INCUBATION_DAYS + 1)
            {
                new_infected++;
            }
        }

        if (is_immune(p))
            immune++;
        else
        {
            susceptibility_sum += p->susceptibility;
            susceptible_count++;
        }
    }

    float avg_susceptibility = susceptible_count > 0
                                   ? susceptibility_sum / susceptible_count
                                   : 0.0f;

    printf("\nDay %d Report:\n", day);
    printf("Alive: %d\n", alive);
    printf("Dead: %d\n", dead);
    printf("Infected: %d\n", infected);
    printf("New Infected: %d\n", new_infected);
    printf("Immune: %d\n", immune);
    printf("Average Susceptibility (non-immune): %.3f\n", avg_susceptibility);
}

void print_occupancies_map(Person *population)
{
    printf("\nOccupancy map (number of people per grid cell):\n");
    for (int x = 0; x < W; x++)
    {
        for (int y = 0; y < H; y++)
        {
            printf("%d ", AT(x, y).occupancy);
        }
        printf("\n");
    }

    // printf("\nOccupancy map of Immune persons:\n");
    // for (int x = 0; x < W; x++)
    // {
    //     for (int y = 0; y < H; y++)
    //     {
    //         int count = 0;
    //         for (int i = 0; i < NP; i++)
    //         {
    //             if (population[i].x == x && population[i].y == y && is_immune(&population[i]))
    //             {
    //                 count++;
    //             }
    //         }
    //         printf("%d ", count);
    //     }
    //     printf("\n");
    // }

    // printf("\nOccupancy map of Infected persons:\n");
    // for (int x = 0; x < W; x++)
    // {
    //     for (int y = 0; y < H; y++)
    //     {
    //         int count = 0;
    //         for (int i = 0; i < NP; i++)
    //         {
    //             if (population[i].x == x && population[i].y == y && is_infected(&population[i]))
    //             {
    //                 count++;
    //             }
    //         }
    //         printf("%d ", count);
    //     }
    //     printf("\n");
    // }
    // printf("\nOccupancy map of Susceptible persons:\n");
    // for (int x = 0; x < W; x++)
    // {
    //     for (int y = 0; y < H; y++)
    //     {
    //         int count = 0;
    //         for (int i = 0; i < NP; i++)
    //         {
    //             if (population[i].x == x && population[i].y == y && !is_immune(&population[i]) && !is_infected(&population[i]))
    //             {
    //                 count++;
    //             }
    //         }
    //         printf("%d ", count);
    //     }
    //     printf("\n");
    // }
    printf("\n");
}