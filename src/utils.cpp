#include "utils.h"
#include <stdlib.h>
#include <stdio.h>

namespace gainet {

static void float_new_memory_error()
{
    fprintf(stderr, "Float new memory error - possibly out of CPU RAM \n");
    exit(EXIT_FAILURE);
}

float *float_new_memory(size_t size) 
{
    float *ptr = new float[size]();
    if(ptr == nullptr) {
        float_new_memory_error();
    }
    return ptr;
}

void float_delete_memory(float *ptr) { delete[]ptr; }


} // namespace gainet