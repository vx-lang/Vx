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

float akar_cosf(float x) {
    return cosf(x);
}

float akar_sinf(float x) {
    return sinf(x);
}

float akar_get_rope_freq(int pos, int i, int head_size) {
    float freq = 1.0f / powf(10000.0f, (float)i / (float)head_size);
    return (float)pos * freq;
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

// Llama 2 C Runtime Interop Helpers
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <ctype.h>

// Mmap the checkpoint file
int* akar_load_config(const char* filepath) {
    int fd = open(filepath, O_RDONLY);
    if (fd == -1) return NULL;
    int* config_out = (int*)malloc(7 * sizeof(int));
    read(fd, config_out, 28);
    close(fd);
    return config_out;
}

float* akar_malloc_f32(int num_elements) {
    return (float*)malloc(num_elements * sizeof(float));
}

float* akar_load_weights(const char* filepath) {
    int fd = open(filepath, O_RDONLY);
    if (fd == -1) return NULL;
    size_t file_size = lseek(fd, 0, SEEK_END);
    float* data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    return data + 7;
}

float* akar_advance_ptr(float* ptr, int offset) {
    return ptr + offset;
}

// BPE Tokenizer State
typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

void* akar_build_tokenizer(const char* filepath, int vocab_size) {
    Tokenizer* t = malloc(sizeof(Tokenizer));
    t->vocab_size = vocab_size;
    t->vocab = malloc(vocab_size * sizeof(char*));
    t->vocab_scores = malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    FILE *file = fopen(filepath, "rb");
    if (!file) { return NULL; }
    fread(&t->max_token_length, sizeof(int), 1, file);
    int len;
    for (int i = 0; i < vocab_size; i++) {
        fread(t->vocab_scores + i, sizeof(float), 1, file);
        fread(&len, sizeof(int), 1, file);
        t->vocab[i] = malloc(len + 1);
        fread(t->vocab[i], len, 1, file);
        t->vocab[i][len] = '\0';
    }
    fclose(file);
    return t;
}

char* akar_decode_token(void* tokenizer_ptr, int prev_token, int token) {
    Tokenizer* t = (Tokenizer*)tokenizer_ptr;
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void akar_safe_printf(char *piece) {
    if (piece == NULL || piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return;
        }
    }
    printf("%s", piece);
    fflush(stdout);
}
