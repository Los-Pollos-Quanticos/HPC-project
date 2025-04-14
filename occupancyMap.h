#ifndef OCCUPANCYMAP_H
#define OCCUPANCYMAP_H

#include "./config.h"

typedef struct
{
    int occupancy;    // Number of people in the cell
    Person **persons; // Array of persons in the cell
} Cell;

extern Cell *occupancy_map;
// Macro to access occupancy_map
#define AT(x, y) occupancy_map[(x) * H + (y)]

void removePerson(Person *p);
void addPerson(Person *p, int x, int y);
void movePerson(Person *p, int x, int y);
void freeOccupancyMap();

#endif
