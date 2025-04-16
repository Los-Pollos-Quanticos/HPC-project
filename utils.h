#ifndef UTILS_H
#define UTILS_H

#include "./config.h"
#include "./occupancyMap.h"

float gaussian_random(float mean, float stddev);
void print_grid(Person *population);

bool is_dead(const Person *p);
bool is_immune(const Person *p);
bool is_infected(const Person *p);
bool is_newly_infected(const Person *p);

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
#endif
