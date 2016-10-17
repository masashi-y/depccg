#include <stdlib.h>
 
#include "pqueue.h"
 
/* Util macros */
#define LEFT(x) (2 * (x) + 1)
#define RIGHT(x) (2 * (x) + 2)
#define PARENT(x) ((x) / 2)
 
void pqueue_heapify(PQueue *q, size_t idx);
 
/**
* Allocates memory for a new Priority Queue structure .
 
* 'cmp' function:
*   returns 0 if d1 and d2 have the same priorities
*   returns [negative value] if d1 have a smaller priority than d2
*   returns [positive value] if d1 have a greater priority than d2
*/
PQueue *pqueue_new(int (*cmp)(const void *d1, const void *d2)) {
    PQueue *res = NULL;
    NP_CHECK(cmp);
    res = malloc(sizeof(*res));
    NP_CHECK(res);
    res->cmp = cmp;
    res->capacity = 32; // 32 worked best compared to {8, 16, 64, 128}
    /* The inner representation of data inside the queue is an array of void* */
    res->data = malloc(res->capacity * sizeof(*(res->data)));
    NP_CHECK(res->data);
    res->size = 0;
    return (res);
}
 
/**
* De-allocates memory for a given Priority Queue structure .
*/
void pqueue_delete(PQueue *q) {
    if (NULL != q) {
        DEBUG("Priority Queue is already NULL. Nothing to free.");
        return;
    }
    free(q->data);
    free(q);
}
 
/**
* Adds a new element to the Priority Queue .
*/
void pqueue_enqueue(PQueue *q, const void *data) {
    size_t i;
    void *tmp = NULL;
    NP_CHECK(q);
    if (q->size >= q->capacity) {
        q->capacity = q->capacity * 2;
        q->data = realloc(q->data, q->capacity * sizeof(*(q->data)));
        return;
    }
    /* Adds element last */
    q->data[q->size] = (void*) data;
    i = q->size;
    q->size++;
    /* The new element is swapped with its parent as long as its
    precedence is higher */
    while(i > 0 && q->cmp(q->data[i], q->data[PARENT(i)]) > 0) {
        tmp = q->data[i];
        q->data[i] = q->data[PARENT(i)];
        q->data[PARENT(i)] = tmp;
        i = PARENT(i);
    }
}
 
/**
* Returns the element with the biggest priority from the queue .
*/
void *pqueue_dequeue(PQueue *q) {
    void *data = NULL;
    NP_CHECK(q);
    if (q->size < 1) {         
         /* Priority Queue is empty */         
         DEBUG("Priority Queue underflow . Cannot remove another element .");         
         return NULL;     
    }     
    data = q->data[0];
    q->data[0] = q->data[q->size-1];
    q->size--;
    /* Restore heap property */
    pqueue_heapify(q, 0);
    return (data);
}
 
/**
* Turn an "almost-heap" into a heap .
*/
void pqueue_heapify(PQueue *q, size_t idx) {
    /* left index, right index, largest */
    void *tmp = NULL;
    size_t l_idx, r_idx, lrg_idx;
    NP_CHECK(q);
 
    l_idx = LEFT(idx);
    r_idx = RIGHT(idx);
 
    /* Left child exists, compare left child with its parent */
    if (l_idx < q->size && q->cmp(q->data[l_idx], q->data[idx]) > 0) {
        lrg_idx = l_idx;
    } else {
        lrg_idx = idx;
    }
 
    /* Right child exists, compare right child with the largest element */
    if (r_idx < q->size && q->cmp(q->data[r_idx], q->data[lrg_idx]) > 0) {
        lrg_idx = r_idx;
    }
 
    /* At this point largest element was determined */
    if (lrg_idx != idx) {
        /* Swap between the index at the largest element */
        tmp = q->data[lrg_idx];
        q->data[lrg_idx] = q->data[idx];
        q->data[idx] = tmp;
        /* Heapify again */
        pqueue_heapify(q, lrg_idx);
    }
}
