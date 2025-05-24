#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Constants for simulation parameters
#define NP 100            // Number of persons
#define INFP 0.05         // Initial percentage of infected persons
#define IMM 0.01          // Initial percentage of immune persons
#define S_AVG 0.5         // Susceptibility average
#define W 100             // Width of the grid
#define H 100             // Height of the grid
#define ND 1              // Number of days in simulation
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

typedef struct
{
    int occupancy;    // Number of people in the cell
    Person **persons; // Array of persons in the cell
} Cell;

Cell *occupancy_map = NULL;
#define AT(x, y) occupancy_map[(x) * H + (y)]

typedef struct
{
    int x;
    int y;
} Tuple;

typedef struct
{
    Tuple *data;  // Array of tuples
    int size;     // Current number of valid tuples
    int capacity; // Maximum capacity of the array
} TList;

TList *createTList(int capacity)
{
    TList *arr = (TList *)malloc(sizeof(TList));
    arr->data = (Tuple *)malloc(sizeof(Tuple) * capacity);
    arr->size = 0;
    arr->capacity = capacity;
    return arr;
}

void addTuple(TList *arr, int x, int y)
{
    if (arr->size >= arr->capacity)
    {
        printf("Array is full!\n");
        return;
    }
    arr->data[arr->size++] = (Tuple){x, y};
}

void removeTupleAt(TList *arr, int index)
{
    if (index < 0 || index >= arr->size)
    {
        printf("Invalid index\n");
        return;
    }
    arr->data[index] = arr->data[arr->size - 1];
    arr->size--;
}

int getRandomTupleIndex(unsigned int seed, TList *arr, Tuple *out)
{
    if (arr->size == 0)
        return -1;

    int idx = rand_r(&seed) % arr->size;
    *out = arr->data[idx];
    return idx;
}

void freeTList(TList *list)
{
    if (list == NULL)
        return;

    if (list->data != NULL)
        free(list->data);

    free(list);
}

void addPerson(Person *p, int x, int y)
{
    AT(x, y).persons[AT(x, y).occupancy] = p;
    AT(x, y).occupancy++;
}

void removePerson(Person *p)
{
    int x = p->x;
    int y = p->y;

    for (int i = 0; i < AT(x, y).occupancy; i++)
    {
        if (AT(x, y).persons[i] == p)
        {
            AT(x, y).occupancy--;

            if (AT(x, y).occupancy == 0 || i == AT(x, y).occupancy)
            {
                AT(x, y).persons[i] = NULL;
            }
            else
            {
                AT(x, y).persons[i] = AT(x, y).persons[AT(x, y).occupancy];
                AT(x, y).persons[AT(x, y).occupancy] = NULL;
            }
            break;
        }
    }
}

void movePerson(Person *p, int x, int y)
{
    removePerson(p);
    addPerson(p, x, y);
}

void freeOccupancyMap()
{
    long grid_size = (long)W * H;
    for (long i = 0; i < grid_size; i++)
    {
        free(occupancy_map[i].persons);
    }
    free(occupancy_map);
}

// CUDA Kernel function
__global__ void hello_world()
{
    printf("Hello World from GPU thread %d of block %d\n", threadIdx.x, blockIdx.x);
}

void init_population(Person *population)
{
    int grid_size = (int)W * H;
    int grid_capacity = (int)grid_size * MAXP_CELL;

    if (NP > grid_capacity)
    {
        printf("Error: Population size exceeds available space on the grid.\n");
        exit(1); // Exits the program if the population size exceeds capacity
    }

    int i;
    int num_immune = (int)(NP * IMM);
    int num_infected = (int)(NP * INFP);

    occupancy_map = (Cell *)malloc(grid_size * sizeof(Cell));

    if (occupancy_map == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for occupancy map.\n");
        return;
    }

    TList *available_coords = createTList(grid_size);
    for (int x = 0; x < W; x++)
    {
        for (int y = 0; y < H; y++)
        {
            addTuple(available_coords, x, y);
            AT(x, y).occupancy = 0;
            AT(x, y).persons = (Person **)malloc(MAXP_CELL * sizeof(Person *));
            if (AT(x, y).persons == NULL)
            {
                fprintf(stderr, "Failed to allocate persons array for cell %d\n", x * H + y);
                exit(1);
            }
        }
    }
}

int main()
{
    Person *population = (Person *)malloc(NP * sizeof(Person));

    if (population == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    init_population(population);

    // Launch the kernel with 2 blocks, each with 4 threads
    hello_world<<<2, 4>>>();

    // Wait for GPU to finish before exiting
    cudaDeviceSynchronize();

    return 0;
}