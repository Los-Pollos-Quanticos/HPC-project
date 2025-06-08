#ifndef UTILS_H
#define UTILS_H

#include "../config.h"
#include "../structures/occupancyMap.h"

#define LOCK(x, y) cell_locks[x * H + y]

float gaussian_random(unsigned int seed, float mean, float stddev);
void print_grid(Person *population);

bool is_dead(const Person *p);
bool is_immune(const Person *p);
bool is_infected(const Person *p);
bool is_newly_infected(const Person *p);
bool is_newly_recovered(const Person *p);

typedef enum
{
    IMMUNE = 0,
    INFECTED = 1,
    SUSCEPTIBLE = 2,
    DEAD = 3
} State;

typedef struct
{
    int x, y;
    State state;
} PersonReport;

void save_population(Person *population, int day);
long get_time_in_ms(struct timespec start, struct timespec end);
#endif
