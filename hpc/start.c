#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

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

// Structure to represent each person
struct Person
{
    float x, y;           // Position on the grid
    int state;            // 0 = susceptible, 1 = infected, 2 = recovered, 3 = dead
    float susceptibility; // Susceptibility to infection
    int immune;           // 0 = non-immune, 1 = immune
    int incubation;       // Incubation days left
};

// Check for CUDA errors
inline void checkCudaErrors(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

// CUDA kernel for moving persons randomly within the grid
__global__ void movePersons(Person *population, int NP, int W, int H)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NP && population[idx].state != 3)
    {
        // Randomly move the person by +/- 1 in the x and y directions
        population[idx].x += (rand() % 3) - 1;
        population[idx].y += (rand() % 3) - 1;

        // Apply toroidal wrapping for x and y coordinates
        population[idx].x = fmodf(population[idx].x + W, W);
        population[idx].y = fmodf(population[idx].y + H, H);
    }
}

// CUDA kernel for spreading infection based on proximity and infection radius (Manhattan distance)
__global__ void infectionSpread(Person *population, int NP, int infectionRadius)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < NP && population[idx].state == 1) // Only process infected individuals
    {
        for (int j = 0; j < NP; ++j)
        {
            // Check if the target person is susceptible or recovered but not immune
            if (j != idx && (population[j].state == 0 || (population[j].state == 2 && population[j].immune == 0)))
            {
                // Calculate the distance in grid terms
                int dx = abs(population[j].x - population[idx].x);
                int dy = abs(population[j].y - population[idx].y);

                // Check for toroidal wrapping (adjust distances for grid)
                if (dx > W / 2)
                    dx = W - dx;
                if (dy > H / 2)
                    dy = H - dy;

                // Calculate the Chebyshev distance
                int distance = max(dx, dy);

                // Check if within infection radius
                if (distance <= infectionRadius)
                {
                    // Probability of infection
                    float infectionChance = beta * population[idx].susceptibility;
                    if (infectionChance > Ith)
                    {
                        population[j].state = 1;
                        population[j].incubation = INCUBATION_DAYS;
                    }
                }
            }
        }
    }
}

// CUDA kernel for handling recovery or death after the incubation period
__global__ void recoveryDeath(Person *population, int NP, float mu)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NP && population[idx].state == 1)
    {
        if (population[idx].incubation > 0)
        {
            population[idx].incubation--;
        }
        else
        {
            float randVal = (float)rand() / RAND_MAX;
            population[idx].state = (randVal < mu) ? 2 : 3;
            population[idx].immune = (randVal < mu) ? (rand() % 2 == 0) : 0;
        }
    }
}

// Function to generate a random number from a beta distribution
__device__ float betaDistribution(float alpha, float beta)
{
    // TODO - use a beta distribution to generate with mean S_effective IKD for the spread factor
}

// CUDA kernel to initialize the population
__global__ void initializePopulationKernel(Person *population, int NP, float S_effective, float INFP, float Imm)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NP)
    {
        // Place people uniformly in the grid
        population[idx].x = (idx % (int)sqrt(NP)) * (W / sqrt(NP));
        population[idx].y = (idx / (int)sqrt(NP)) * (H / sqrt(NP));

        // TODO - Initialize the state, susceptibility, and immune status of each person
    }
}

// Function to initialize the population
void initializePopulation(Person *population, int NP, float S_effective, float INFP, float Imm)
{
    Person *d_population;
    checkCudaErrors(cudaMalloc(&d_population, NP * sizeof(Person)));

    int blockSize = 256;
    int numBlocks = (NP + blockSize - 1) / blockSize;

    initializePopulationKernel<<<numBlocks, blockSize>>>(d_population, NP, S_effective, INFP, Imm);
    checkCudaErrors(cudaMemcpy(population, d_population, NP * sizeof(Person), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_population));
}

// Main simulation function
int main()
{
    float S_effective = S_AVG / (1 - IMM);
    Person *h_population = (Person *)malloc(NP * sizeof(Person));

    initializePopulation(h_population, NP, S_effective, INFP, IMM);

    Person *d_population;
    checkCudaErrors(cudaMalloc(&d_population, NP * sizeof(Person)));
    checkCudaErrors(cudaMemcpy(d_population, h_population, NP * sizeof(Person), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (NP + blockSize - 1) / blockSize;

    for (int day = 0; day < ND; day++)
    {
        movePersons<<<numBlocks, blockSize>>>(d_population, NP, W, H);
        infectionSpread<<<numBlocks, blockSize>>>(d_population, NP, BETA, ITH, IRD);
        recoveryDeath<<<numBlocks, blockSize>>>(d_population, NP, MU);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    checkCudaErrors(cudaMemcpy(h_population, d_population, NP * sizeof(Person), cudaMemcpyDeviceToHost));

    // Analyze results
    int susceptible = 0, infected = 0, recovered = 0, dead = 0;
    for (int i = 0; i < NP; i++)
    {
        if (h_population[i].state == 0)
            susceptible++;
        else if (h_population[i].state == 1)
            infected++;
        else if (h_population[i].state == 2)
            recovered++;
        else if (h_population[i].state == 3)
            dead++;
    }

    printf("Susceptible: %d\n", susceptible);
    printf("Infected: %d\n", infected);
    printf("Recovered: %d\n", recovered);
    printf("Dead: %d\n", dead);

    checkCudaErrors(cudaFree(d_population));
    free(h_population);

    return 0;
}
