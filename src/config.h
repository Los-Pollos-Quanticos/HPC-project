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
#ifndef W
#define W 3 // Width of the grid
#endif

#ifndef H
#define H 3 // Height of the grid
#endif

#ifndef MAXP_CELL
#define MAXP_CELL 3 // Maximum number of people in a cell
#endif

#ifndef NP
#define NP (int)(W * H * MAXP_CELL * 0.1) // Total number of people in the simulation
#endif

#ifndef INFP
#define INFP 0.5f // Initial percentage of infected persons
#endif

#ifndef IMM
#define IMM 0.1f // Initial percentage of immune persons
#endif

#ifndef S_AVG
#define S_AVG 0.5f // Susceptibility average
#endif

#ifndef ND
#define ND 20 // Number of days in simulation
#endif

#ifndef INCUBATION_DAYS
#define INCUBATION_DAYS 4 // Incubation period in days
#endif

#ifndef BETA
#define BETA 0.8f // Contagiousness factor
#endif

#ifndef ITH
#define ITH 0.2f // Infection threshold
#endif

#ifndef IRD
#define IRD 1 // Infection radius (in cells)
#endif

#ifndef MU
#define MU 0.6f // Probability of recovery after infection
#endif

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
