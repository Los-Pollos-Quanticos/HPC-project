#ifndef UTILS_H
#define UTILS_H

#include "./config.h"
#include "./occupancyMap.h"

float gaussian_random(float mean, float stddev);
void print_grid(Person *population);
void stats(Person *population);
void print_occupancies_map(Person *population);
void print_daily_report(Person *population, int day);

bool is_dead(const Person *p);
bool is_immune(const Person *p);
bool is_infected(const Person *p);
#endif
