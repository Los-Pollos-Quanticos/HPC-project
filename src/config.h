#ifndef CONFIG_H
#define CONFIG_H
#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <string.h>

// Constants for simulation parameters
#define W 10000     // Width of the grid
#define H 10000     // Height of the grid
#define MAXP_CELL 3 // Maximum number of people in a cell
#define NP 10000000
#define INFP 0.5f         // Initial percentage of infected persons
#define IMM 0.1f          // Initial percentage of immune persons
#define S_AVG 0.5f        // Susceptibility average
#define ND 20             // Number of days in simulation
#define INCUBATION_DAYS 4 // Incubation period in days
#define BETA 0.8f         // Contagiousness factor
#define ITH 0.2f          // Infection threshold
#define IRD 1             // Infection radius (in cells)
#define MU 0.6f           // Probability of recovery after infection

#define PI 3.14159265358979323846

    // Structs
    typedef struct
    {
        int x, y;             // Position in the grid, if x < 0 or y < 0, the person is dead
        float susceptibility; // From 0 to 1, if 0 = immune otherwise susceptible
        int incubation_days;  // is 0 if not infected, > 0 if infected
        bool new_infected;    // is true if the person was infected within day D
    } Person;

#ifdef __cplusplus
}
#endif
#endif
