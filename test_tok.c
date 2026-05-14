#include <stdio.h>
#include <stdlib.h>

int main() {
    FILE *file = fopen("tokenizer.bin", "rb");
    int max_token_length;
    fread(&max_token_length, sizeof(int), 1, file);
    for (int i = 0; i < 300; i++) {
        float score;
        fread(&score, sizeof(float), 1, file);
        int len;
        fread(&len, sizeof(int), 1, file);
        char* word = malloc(len + 1);
        fread(word, len, 1, file);
        word[len] = '\0';
        printf("Token %d: '%s' (len %d)\n", i, word, len);
        free(word);
    }
    fclose(file);
    return 0;
}
