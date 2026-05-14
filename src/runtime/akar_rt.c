#include <stdio.h>
#include <stdint.h>

int64_t akar_transfer(int64_t tensor_id) {
    printf("[Akar Hardware Mock] Transferring tensor ID: %lld via DMA engine...\n", tensor_id);
    return tensor_id; // Just return the ID for now
}

int64_t custom_matmul(int64_t tensor_a, int64_t tensor_b) {
    printf("[Akar Hardware Mock] Executing custom_matmul on NPU for tensor IDs: %lld and %lld...\n", tensor_a, tensor_b);
    return tensor_a + tensor_b; // Return a modified dummy ID based on both inputs
}

void akar_print(int64_t tensor_id) {
    printf("[Akar Runtime] Computation finished! Final tensor ID: %lld\n", tensor_id);
}
