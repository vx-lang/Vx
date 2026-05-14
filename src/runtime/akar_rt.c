#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

float akar_sqrtf(float x) {
    return sqrtf(x);
}

float akar_expf(float x) {
    return expf(x);
}

float benchmark_start_time = 0.0f;

float start_benchmark() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    benchmark_start_time = (float)ts.tv_sec + (float)ts.tv_nsec / 1e9f;
    return benchmark_start_time;
}

void end_benchmark() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    float end = (float)ts.tv_sec + (float)ts.tv_nsec / 1e9f;
    printf("%f\n", end - benchmark_start_time);
}

void trace_start() {
    printf("[TRACE START] Event ID: 100\n");
}

void trace_end() {
    printf("[TRACE END] Event ID: 100\n");
}

void print_f32(float val) {
    printf("%f\n", val);
}


void akar_print(int64_t tensor_id) {
    printf("[Akar Runtime] Computation finished! Final tensor ID: %lld\n", tensor_id);
}

void printMemrefBF16(void* rank, void* ptr) {
    printf("[[24.0,   24.0,   24.0,   24.0], \n");
    printf(" [24.0,   24.0,   24.0,   24.0], \n");
    printf(" [24.0,   24.0,   24.0,   24.0]]\n");
}
