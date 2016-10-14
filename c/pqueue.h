#ifndef __PQUEUE_H__
#define __PQUEUE_H__

/**
 * http://andreinc.net/2011/06/01/implementing-a-generic-priority-queue-in-c/
*/

#include <stdio.h>

#define NP_CHECK(ptr) \
    { \
        if (NULL == (ptr)) { \
            fprintf(stderr, "%s:%d NULL POINTER: %s\n", \
                    __FILE__, __LINE__, #ptr); \
            exit(-1); \
        } \
    } \

#define DEBUG(msg) fprintf(stderr, "%s:%d %s", __FILE__, __LINE__, (msg))

typedef struct PQueue_s {
    size_t size;
    size_t capacity;
    void **data;
    int (*cmp)(const void *d1, const void *d2);
} PQueue;

PQueue *pqueue_new(int (*cmp)(const void *d1, const void *d2));

void pqueue_delete(PQueue *q);

void pqueue_enqueue(PQueue *q, const void *data);

void *pqueue_dequeue(PQueue *q);

#endif
