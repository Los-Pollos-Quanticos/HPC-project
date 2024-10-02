#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

//PROVO A INSERIRE COMMENTO FEDERICO MUSCARà

/**Chiedere al prof:
 * Susceptibility, come distribuirla. Questo risulta più limitante. Gli immuni sono gli unici che hanno S<Sth?
 *                 come si tiene in considerazione di Savg?
 * Movimento circolare delle persone
 * Distribuzione omogenea iniziale delle persone, io ho fatto così, per Marco va bene distribuzione random
 */

typedef struct
{
    int x, y;             // Position in the grid
    float susceptibility; // From 0 to 1
    int state;            // 0:susceptible ; 1:immune, 2:infected ; 3:recovered ; 4:dead
    int incubation_days;
} Person;

// Constants for simulation parameters
#define NP 1000000        // Number of persons
#define INFP 0.05         // Initial percentage of infected persons
#define IMM 0.01          // Initial percentage of immune persons
#define S_AVG 0.5         // Susceptibility average
#define W 10000           // Width of the grid
#define H 4500            // Height of the grid
#define ND 200            // Number of days in simulation
#define INCUBATION_DAYS 4 // Incubation period in days
#define BETA 0.8          // Contagiousness factor
#define ITH 0.5           // Infection threshold
#define IRD 1.0           // Infection radius (1 meter)
#define MU 0.6            // Probability of recovery after infection

#define M_PI 3.14159265358979323846

// Random float generator in [min, max]
float random_float()
{
    return ((float)rand() / RAND_MAX);
}

// Gaussian random number through Box-Muller transform
float gaussian_random(float mean, float stddev)
{
    float u = ((float)rand() / RAND_MAX);
    float v = ((float)rand() / RAND_MAX);
    return mean + stddev * sqrt(-2.0f * log(u)) * cos(2.0f * M_PI * v);
}

// Function to assign susceptibility greater than Sth for susceptible
// and infected people following the Gaussian distribution
float assign_susceptibility(float mean, float stddev, float Sth)
{
    float susceptibility;
    do
    {
        susceptibility = gaussian_random(mean, stddev);
    } while (susceptibility < Sth); // To ensure it's above the threshold
    return susceptibility;
}

// Distribute uniformly the population over the entire grid
void population_distribution(Person *population)
{
    for (int i = 0; i < NP; i++)
    {
        // Population uniform distribution
        population[i].x = (i % (int)sqrt(NP)) * (W / sqrt(NP));
        population[i].y = (i / (int)sqrt(NP)) * (H / sqrt(NP));
    }
}

// After the population has been distributed: assignment of status to each person
void population_initialization(Person *population)
{
    int num_immune = (int)(NP * IMM);      // Number of immune individuals
    int num_infectious = (int)(NP * INFP); // Number of initially infectious individuals
    float Sth = (float)(ITH / BETA);       // Threshold susceptibility

    // Seed for random number generation
    srand(time(NULL));

    // Randomly shuffle the population to avoid biases in assignment
    for (int i = NP - 1; i >= 0; i--)
    {
        int j = rand() % (i + 1);
        Person temp = population[i];
        population[i] = population[j];
        population[j] = temp;
    }

    // Assign status and susceptibility to the shuffled population
    for (int i = 0; i < NP; i++)
    {
        if (i < num_immune)
        {
            // Immune individuals, susceptibility = 0
            population[i].state = 1;
            population[i].susceptibility = 0;
        }
        else if (i < num_immune + num_infectious)
        {
            // Infectious individuals
            population[i].state = 2;
            population[i].incubation_days = INCUBATION_DAYS;
        }
        else
        {
            // Susceptible individuals
            population[i].state = 0;
        }

        // For susceptible and infectious individuals, assign susceptibility
        if (population[i].state == 0 || population[i].state == 2)
        {
            population[i].susceptibility = assign_susceptibility(S_AVG, 0.1, Sth);
            // Ensure susceptibility does not exceed 1
            if (population[i].susceptibility > 1)
            {
                population[i].susceptibility = 1;
            }
        }
    }
}

// Update position with toroidal wrapping
void move_population(Person *population)
{
    for (int i = 0; i < NP; i++)
    {
        // Move randomly by 1 step in any direction
        int dx = (rand() % 3) - 1; // -1, 0, or 1
        int dy = (rand() % 3) - 1; // -1, 0, or 1

        // Update position and apply toroidal wrapping
        population[i].x = (int)(population[i].x + dx + W) % W;
        population[i].y = (int)(population[i].y + dy + H) % H;
    }
}

// Not the best but the first that comes in my mind...
//  Check for infections within infection radius (with toroidal wrapping)
void check_infections(Person *population)
{
    for (int i = 0; i < NP; i++)
    {
        if (population[i].state == 0)
        {
            for (int j = 0; j < NP; j++)
            {
                if (population[j].state == 2)
                {
                    // Calculate distance with toroidal wrapping
                    int dx = abs(population[i].x - population[j].x);
                    int dy = abs(population[i].y - population[j].y);

                    // Wrap the distance across the grid
                    if (dx > W / 2)
                        dx = W - dx;
                    if (dy > H / 2)
                        dy = H - dy;

                    // Check if within infection radius
                    if (dx <= IRD && dy <= IRD)
                    {
                        // Calculate infection probability
                        float infection_prob = BETA * population[i].susceptibility;
                        if (infection_prob >= ITH)
                        {
                            // Infect the person: buon recupero!
                            population[i].state = 2;
                            population[i].incubation_days = INCUBATION_DAYS;
                            break;
                        }
                    }
                }
            }
        }
    }
}

// Simulate disease progression (recovery or death after incubation)
void update_disease_progression(Person *population)
{
    for (int i = 0; i < NP; i++)
    {
        if (population[i].state == 2)
        {
            population[i].incubation_days -= 1;
            // Check if incubation period has passed
            if (population[i].incubation_days == 0)
            {
                // After incubation, recover or die
                if (random_float(0, 1) <= MU)
                {
                    population[i].state = 3;
                    population[i].incubation_days = 4;
                    if (random_float(0, 1) < 0.5)
                    {
                        population[i].state = 1; // Person has become as Marco with respect to the 29 mark.
                    }
                }
                else
                {
                    population[i].state = 4; // Person has sadly passed away rip ciaone
                }
            }
        }
    }
}

int main()
{
    Person *population = (Person *)malloc(NP * sizeof(Person));

    population_distribution(population);
    population_initialization(population);

    for (int day = 0; day < ND; day++)
    {
        move_population(population);
        check_infections(population);
        update_disease_progression(population);
    }

    free(population);
    return 0;
}
