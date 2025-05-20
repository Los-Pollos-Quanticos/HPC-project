#ifndef CONFIG_H
#define CONFIG_H

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <limits.h>

// Constants for simulation parameters
#define NP 100000        // Number of persons
#define INFP 0.05         // Initial percentage of infected persons
#define IMM 0.01          // Initial percentage of immune persons
#define S_AVG 0.5         // Susceptibility average
#define W 700            // Width of the grid
#define H 200           // Height of the grid
#define ND 50            // Number of days in simulation
#define INCUBATION_DAYS 4 // Incubation period in days
#define BETA 0.8          // Contagiousness factor
#define ITH 0.5           // Infection threshold
#define IRD 1.0           // Infection radius (1 meter)
#define MU 0.6            // Probability of recovery after infection
#define MAXP_CELL 3       // Maximum number of people in a cell

#define PI 3.14159265358979323846

// Structs
typedef struct
{
    int x, y;             // Position in the grid, if x < 0 or y < 0, the person is dead
    float susceptibility; // From 0 to 1, if 0 = immune otherwise susceptible
    int incubation_days;  // is 0 if not infected, > 0 if infected
    bool new_infected;    // is true if the person was infected within day D
} Person;
#endif
