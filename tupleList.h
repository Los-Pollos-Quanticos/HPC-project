#ifndef TUPLELIST_H
#define TUPLELIST_H
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

TList *createTList(int capacity);

void addTuple(TList *arr, int x, int y);

void removeTupleAt(TList *arr, int index);

int getRandomTupleIndex(unsigned int seed, TList *arr, Tuple *out);

void freeTList(TList *list);
#endif