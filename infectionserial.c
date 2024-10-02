#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct{
    int x, y; //Position on the grid
    float susceptibility; //From 0 to 1
    int state;  //0:susceptible ; 1:immune, 2:infected ; 3:dead
    int incubation_days;
    bool recovered;
}Person;

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
#define STDDEV 0.1

// Random float generator in [0, 1]
float random_float() {
    return ((float)rand() / RAND_MAX);
}

// Gaussian random number through Box-Muller transform
float gaussian_random(float mean, float stddev) {
    float u = random_float();
    float v = random_float();
    return mean + stddev * sqrt(-2.0f * log(u)) * cos(2.0f * M_PI * v);
}

float assign_susceptibility(float mean, float stddev) {
    //This funciton is called only for state =0,2
    float susceptibility = gaussian_random(mean, stddev); //Generate the susceptibility of the person as a Gaussian using parameters (mean, stddev)
    return susceptibility;
}

//After the population has been distributed: assignment of status to each person

void population_initialization(Person population[], float stddev) {
    srand(time(NULL));

    int num_immune = (int)(NP * IMM);       // Number of immune individuals
    int num_infectious = (int)(NP * INFP);  // Number of initially infectious individuals
    float Sth = (float)(ITH / BETA);        // Threshold susceptibility

    // Loop to initialize the population
    for (int i = 0; i < NP; i++) {
        if (i < num_immune) {
            population[i].state = 1;            // Immune
            population[i].susceptibility = 0.0; // S = 0
        } else if (i >= num_immune && i < num_immune + num_infectious) {
            population[i].state = 2;                  // Infectious
            population[i].incubation_days = INCUBATION_DAYS; // Incubation period
            population[i].susceptibility = assign_susceptibility(S_AVG, stddev);
        } else {
            population[i].state = 0; // Susceptible
            population[i].susceptibility = assign_susceptibility(S_AVG, stddev);
        }

        // Random position on the grid
        int row = (int)(H * random_float()); // A random number between 0 and (H-1)
        int col = (int)(W * random_float()); // A random number between 0 and (W-1)
        population[i].x = col;
        population[i].y = row;
    }
}

void move_population(Person* population) {
    for (int i=0; i<NP; i++) {
        // Move randomly by 1 step in any direction
        int dx = (rand() % 3) - 1;  // -1, 0, or 1
        int dy = (rand() % 3) - 1;  // -1, 0, or 1
        //This code allows for dx =dy = 0; is it fine? (use a do while until (dx == 0 && dy == 0))

        //Non mi fa impazzire il toridal wrapping
        // Check if movement would cause out-of-bounds for X
        if (population[i].x + dx >= 0 && population[i].x + dx < W) {
            population[i].x += dx;
        }
        //Sistema con do while

        // Check if movement would cause out-of-bounds for Y
        if (population[i].y + dy >= 0 && population[i].y + dy < H) {
            population[i].y += dy;
        }
    }
}


// CHECK FOR INFECTIONS WITHIN RADIUS (NO WRAPPING) 
void check_infections(Person* population){
    int radius = IRD; //Radius

    for (int i = 0; i < NP; i++) {
        if (population[i].state == 2) {
            population[i].incubation_days = population[i].incubation_days -1;
    }
    
    for (int i = 0; i < NP; i++) {
        if (population[i].state == 0) {
            for (int j = 0; j < NP; j++) {
                if (population[j].state == 2 && population[i].incubation_days != INCUBATION_DAYS) {//Discard the NEWLY INFECTED!
                    // Calculate distance 
                    int dx = abs(population[i].x - population[j].x);
                    int dy = abs(population[i].y - population[j].y);

                    // Check if within infection radius
                    if (dx <= IRD && dy <= IRD) {
                        // Calculate infection probability
                        float infection_prob = BETA * population[i].susceptibility;
                        if (infection_prob >= IRD) {
                            // Infect the person: buon recupero!
                            population[i].state = 2;
                            population[i].incubation_days = INCUBATION_DAYS;
                        }
                    }
                }
            }
        }
    }
}
}

void update_disease_progression(Person* population) {
    for (int i = 0; i < NP; i++) {
        if (population[i].state == 2 && population[i].incubation_days==0) {
            // Check if incubation period has passed
            if (population[i].incubation_days == 0) {
                // After incubation, recover or die
                if (random_float() <= MU) {
                    population[i].recovered = 0; 
                    //population[i].incubation_days = 4;
                    if (random_float() < 0.5) {
                        population[i].state = 1;  
                        //population[i].susceptibility = 0;
                    }
                } else {
                    population[i].state = 4; //GODO 
                }
            }
        }
    }
}






int main() {
    float stddev = STDDEV;                   // Standard deviation for susceptibility
    Person* population = (Person*)malloc(NP * sizeof(Person));
    if (population == NULL) {
        printf("Memory allocation failed!\n");
        return 1; // Exit if memory allocation fails
    }

    population_initialization(population, stddev); // Initialize the population
    int num_imm=0;
    int num_malati = 0;

    // Print a few members of the population to verify initialization
    for (int i = 0; i < NP; i++) {           // Print the first 10 persons
    if(population[i].state==1)
        num_imm+=1;
    
    if(population[i].state==2)
        num_malati+=1;
    
        printf("Person number: %d, state: %d, susceptibility: %f, position[%d][%d]\n", 
                i, population[i].state, population[i].susceptibility, 
                population[i].x, population[i].y);
    }
    printf("Immuni: %d, Malati: %d", num_imm, num_malati);
    return 0;
}