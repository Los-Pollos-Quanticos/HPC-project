#include "./utils.h"

// Gaussian random number through Box-Muller transform
// TODO: check that the distribution is a gaussian one
float gaussian_random(unsigned int seed, float mean, float stddev)
{
    float u = ((float)rand_r(&seed) / RAND_MAX);
    float v = ((float)rand_r(&seed) / RAND_MAX);
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
    return p->incubation_days > 0 && !is_dead(p);
}

bool is_newly_infected(const Person *p)
{
    return p->new_infected;
}

void save_population(Person *population, int day)
{
    char filename[32];
    snprintf(filename, sizeof(filename), "./report/day_%03d.dat", day);

    FILE *f = fopen(filename, "wb");
    if (!f)
        return;

    int np_value = NP;
    fwrite(&np_value, sizeof(int), 1, f);
    for (int i = 0; i < NP; i++)
    {
        int state;
        if (is_dead(&population[i]))
            state = DEAD;
        else if (is_immune(&population[i]))
            state = IMMUNE;
        else if (is_infected(&population[i]))
            state = INFECTED;
        else
            state = SUSCEPTIBLE;

        PersonReport out = {
            .x = population[i].x,
            .y = population[i].y,
            .state = state};
        fwrite(&out, sizeof(PersonReport), 1, f);
    }

    fclose(f);
}