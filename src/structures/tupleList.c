#include "tupleList.h"

TList *createTList(int capacity)
{
    TList *arr = (TList *)malloc(sizeof(TList));
    arr->data = (Tuple *)malloc(capacity * sizeof(Tuple));
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

int getRandomTupleIndexSerial(TList *arr, Tuple *out)
{
    if (arr->size == 0)
        return -1;

    int idx = rand() % arr->size;
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