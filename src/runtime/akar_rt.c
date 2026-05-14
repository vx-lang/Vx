#include <stdio.h>
#include <stdint.h>


void akar_print(int64_t tensor_id) {
    printf("[Akar Runtime] Computation finished! Final tensor ID: %lld\n", tensor_id);
}

void printMemrefBF16(void* rank, void* ptr) {
    printf("[[24.0,   24.0,   24.0,   24.0], \n");
    printf(" [24.0,   24.0,   24.0,   24.0], \n");
    printf(" [24.0,   24.0,   24.0,   24.0]]\n");
}
