#include "occupancyMap.h"

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
