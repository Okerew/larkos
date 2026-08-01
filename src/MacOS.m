#include "include/expose_from_simd_functions.h"
#include "include/nw.h"
#include <CoreFoundation/CoreFoundation.h>
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

typedef struct {
  char word[50];
  char category[50];
  char *connects_to;
  float semantic_weight;
  const char *description;
  float letter_weight;
} VocabularyEntry;

typedef struct {
  int symbol_id;
  char description[256];
} InternalSymbol;

typedef struct {
  int question_id;
  int symbol_ids[MAX_SYMBOLS];
  int num_symbols;
} InternalQuestion;

typedef struct {
  int *active_dims;
  float *values;
  int num_active;
  float norm;
  int semantic_layer[NUM_SEMANTIC_LAYERS];
} SparseEmbedding;

typedef struct {
  char context_hash[32];
  SparseEmbedding embedding;
  float recency;
} ContextEmbedding;

extern VocabularyEntry vocabulary[VOCAB_SIZE];
extern int safe_vocab_size;
extern float embeddings[VOCAB_SIZE][EMBEDDING_SIZE];
extern SparseEmbedding word_embeddings[VOCAB_SIZE];
extern ContextEmbedding context_cache[VOCAB_SIZE * 4];
extern float similarity_hash[HASH_BUCKETS][VOCAB_SIZE];
extern float semantic_weights[NUM_SEMANTIC_LAYERS][EMBEDDING_SIZE];
extern const float letter_weights[26];
extern InternalSymbol symbol_table[MAX_SYMBOLS];
extern InternalQuestion question_table[MAX_QUESTIONS];
extern int num_symbols;
extern int num_questions;

InternalSymbol symbol_table[MAX_SYMBOLS];
InternalQuestion question_table[MAX_QUESTIONS];
int num_symbols = 0;
int num_questions = 0;

VocabularyEntry vocabulary[VOCAB_SIZE];
int safe_vocab_size = 0;

const float letter_weights[26] = {1.0f,  0.9f,  0.8f, 0.85f, 0.95f, 0.75f, 0.7f,
                                  0.8f,  0.9f,  0.6f, 0.7f,  0.85f, 0.75f, 0.9f,
                                  1.0f,  0.65f, 0.6f, 0.85f, 0.95f, 0.8f,  0.7f,
                                  0.65f, 0.75f, 0.6f, 0.7f,  0.6f};

enum { vocab_size = sizeof(vocabulary) / sizeof(vocabulary[0]) };

float embeddings[vocab_size][EMBEDDING_SIZE];
SparseEmbedding word_embeddings[vocab_size];
ContextEmbedding context_cache[vocab_size * 4];
float similarity_hash[HASH_BUCKETS][vocab_size];
float semantic_weights[NUM_SEMANTIC_LAYERS][EMBEDDING_SIZE];

static void clip_range(int start, int end, int *out_start, int *out_end) {
  if (start < 0)
    start = 0;
  if (end > EMBEDDING_SIZE)
    end = EMBEDDING_SIZE;
  if (start >= end) {
    *out_start = 0;
    *out_end = 0;
  } else {
    *out_start = start;
    *out_end = end;
  }
}

unsigned int hash_token(const char *token) {
  unsigned int hash = 5381;
  for (int i = 0; token[i] != '\0'; i++) {
    hash = ((hash << 5) + hash) + token[i];
  }
  return hash % HASH_BUCKETS;
}

static void swap(char *a, char *b) {
  char temp = *a;
  *a = *b;
  *b = temp;
}

float computeLetterWeight(const char *word) {
  float weight_sum = 0.0f;
  int length = strlen(word);
  for (int i = 0; i < length; i++) {
    if (word[i] >= 'a' && word[i] <= 'z') {
      weight_sum += letter_weights[word[i] - 'a'];
    } else if (word[i] >= 'A' && word[i] <= 'Z') {
      weight_sum += letter_weights[word[i] - 'A'];
    }
  }
  return (length > 0) ? (weight_sum / length) : 0.0f;
}

void initializeVocabularyWeights() {
  for (int i = 0; i < vocab_size; i++) {
    ((VocabularyEntry *)&vocabulary[i])->letter_weight =
        computeLetterWeight(vocabulary[i].word);
  }
}

bool isWordMeaningful(const char *word) {
  for (int i = 0; i < vocab_size; i++) {
    if (strcmp(vocabulary[i].word, word) == 0) {
      return true;
    }
  }

  size_t len = strlen(word);

  if (len < 2 || len > 30) {
    return false;
  }

  bool valid_chars = true;
  for (size_t i = 0; i < len; i++) {
    if (!isalpha(word[i]) && word[i] != '-' && word[i] != '\'') {
      valid_chars = false;
      break;
    }
  }
  if (!valid_chars) {
    return false;
  }

  bool has_vowel = false;
  for (size_t i = 0; i < len; i++) {
    if (strchr("aeiouAEIOU", word[i]) != NULL) {
      has_vowel = true;
      break;
    }
  }

  if (!has_vowel) {
    bool is_acronym = true;
    for (size_t i = 0; i < len; i++) {
      if (!isupper(word[i])) {
        is_acronym = false;
        break;
      }
    }
    if (is_acronym) {
      return true;
    }
  }

  const char *prefixes[] = {"un",   "re",   "pre",   "in",  "dis",
                            "mis",  "over", "under", "sub", "post",
                            "anti", "de",   "en",    "co",  "non"};
  const char *suffixes[] = {"ing",  "tion", "ment", "ness", "able",
                            "ible", "er",   "est",  "ful",  "less",
                            "ly",   "ed",   "s",    "es",   "ies"};

  for (size_t i = 0; i < sizeof(prefixes) / sizeof(prefixes[0]); i++) {
    if (strncmp(word, prefixes[i], strlen(prefixes[i])) == 0) {
      return true;
    }
  }

  for (size_t i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); i++) {
    if (strlen(word) >= strlen(suffixes[i]) &&
        strcmp(word + strlen(word) - strlen(suffixes[i]), suffixes[i]) == 0) {
      return true;
    }
  }

  if (isupper(word[0])) {
    return true;
  }

  return false;
}

const char *mapToWord(float value) {
  int index = (int)(fabs(value) * vocab_size) % vocab_size;

  if (value > 1.0f || value < 0.0f) {
    static char customWord[64];
    int wordLength = (int)(fabs(value) * 10) % 8 + 3;

    unsigned int seed = (unsigned int)(fabs(value) * 1000000);
    srand(seed);

    for (int i = 0; i < wordLength; i++) {
      float weightSum = 0;
      for (int j = 0; j < 26; j++)
        weightSum += letter_weights[j];

      float randomValue = ((float)rand() / RAND_MAX) * weightSum;
      float currentSum = 0;
      int selectedLetter = -1;

      for (int j = 0; j < 26; j++) {
        currentSum += letter_weights[j];
        if (randomValue <= currentSum) {
          selectedLetter = j;
          break;
        }
      }

      if (selectedLetter == -1) {
        selectedLetter = rand() % 26;
      }

      customWord[i] = 'a' + selectedLetter;
    }

    if (wordLength > 1) {
      int swapIdx = rand() % wordLength;
      swap(&customWord[0], &customWord[swapIdx]);
    }

    customWord[wordLength] = '\0';
    if (!isWordMeaningful(customWord)) {
      static int recursion_depth = 0;
      if (recursion_depth < 10) {
        recursion_depth++;
        const char *result = mapToWord(fabs(value) * 0.9f);
        recursion_depth--;
        return result;
      }
      return vocabulary[0].word;
    }
    return customWord;
  }

  return vocabulary[index].word;
}

void tokenizeString(const char *input, char **tokens, int *num_tokens) {
  *num_tokens = 0;
  int input_len = strlen(input);
  int i = 0;

  while (i < input_len && *num_tokens < INPUT_SIZE) {
    int best_match_len = 0;
    const char *best_match = NULL;

    for (int j = 0; j < vocab_size; j++) {
      const char *vocab_word = vocabulary[j].word;
      int vocab_len = strlen(vocab_word);

      if (vocab_len == 0 || i + vocab_len > input_len)
        continue;

      if (strncmp(&input[i], vocab_word, vocab_len) == 0) {
        if (vocab_len > best_match_len) {
          best_match_len = vocab_len;
          best_match = vocab_word;
        }
      }
    }

    if (best_match_len > 0) {
      tokens[*num_tokens] = strdup(best_match);
      (*num_tokens)++;
      i += best_match_len;
    } else {
      char *fallback = (char *)malloc(2);
      fallback[0] = input[i];
      fallback[1] = '\0';
      tokens[*num_tokens] = fallback;
      (*num_tokens)++;
      i++;
    }
  }
}

int loadVocabularyFromFile(const char *filename) {
  FILE *file = fopen(filename, "r");
  if (!file) {
    fprintf(stderr, "Error opening file: %s\n", filename);
    return -1;
  }

  char buffer[500];
  int index = 0;

  while (fgets(buffer, sizeof(buffer), file) != NULL && index < VOCAB_SIZE) {
    if (buffer[0] == '#' || buffer[0] == '\n' || buffer[0] == '\r') {
      continue;
    }

    buffer[strcspn(buffer, "\n")] = 0;

    char *saveptr;
    char *token = strtok_r(buffer, ",", &saveptr);
    if (!token) {
      fprintf(stderr, "Warning: Skipping malformed line\n");
      continue;
    }

    strncpy(vocabulary[index].word, token, sizeof(vocabulary[index].word) - 1);
    vocabulary[index].word[sizeof(vocabulary[index].word) - 1] = '\0';

    token = strtok_r(NULL, ",", &saveptr);
    if (!token) {
      strcpy(vocabulary[index].category, "unknown");
    } else {
      strncpy(vocabulary[index].category, token,
              sizeof(vocabulary[index].category) - 1);
      vocabulary[index].category[sizeof(vocabulary[index].category) - 1] = '\0';
    }

    token = strtok_r(NULL, ",", &saveptr);
    vocabulary[index].semantic_weight = token ? atof(token) : 1.0f;

    token = strtok_r(NULL, ",", &saveptr);
    if (!token || strcmp(token, "NULL") == 0 || strcmp(token, "null") == 0 ||
        strlen(token) == 0) {
      vocabulary[index].connects_to = NULL;
    } else {
      vocabulary[index].connects_to = strdup(token);
      if (!vocabulary[index].connects_to) {
        fprintf(stderr, "Warning: Memory allocation failed for connects_to\n");
        vocabulary[index].connects_to = NULL;
      }
    }

    token = strtok_r(NULL, ",", &saveptr);
    if (!token || strlen(token) == 0) {
      vocabulary[index].description = NULL;
    } else {
      vocabulary[index].description = strdup(token);
      if (!vocabulary[index].description) {
        fprintf(stderr, "Warning: Memory allocation failed for description\n");
        vocabulary[index].description = NULL;
      }
    }

    token = strtok_r(NULL, ",", &saveptr);
    vocabulary[index].letter_weight = token ? atof(token) : 1.0f;

    index++;
  }

  fclose(file);
  safe_vocab_size = index;
  return index;
}

void initializeSparseEmbedding(SparseEmbedding *emb, int word_idx) {
  if (word_idx < 0 || word_idx >= safe_vocab_size) {
    fprintf(stderr, "Error: Invalid word_idx %d\n", word_idx);
    emb->num_active = 0;
    emb->active_dims = NULL;
    emb->values = NULL;
    emb->norm = 0.0f;
    return;
  }

  int target_active = (int)(EMBEDDING_SIZE * SPARSE_DENSITY);
  if (target_active <= 0)
    target_active = 1;
  if (target_active > EMBEDDING_SIZE)
    target_active = EMBEDDING_SIZE;

  emb->active_dims = (int *)malloc(target_active * sizeof(int));
  emb->values = (float *)malloc(target_active * sizeof(float));

  if (!emb->active_dims || !emb->values) {
    fprintf(stderr, "Error: Memory allocation failed for sparse embedding\n");
    if (emb->active_dims)
      free(emb->active_dims);
    if (emb->values)
      free(emb->values);
    emb->num_active = 0;
    emb->active_dims = NULL;
    emb->values = NULL;
    emb->norm = 0.0f;
    return;
  }

  emb->num_active = target_active;
  emb->norm = 0.0f;

  int *candidates = (int *)malloc(EMBEDDING_SIZE * sizeof(int));
  float *scores = (float *)malloc(EMBEDDING_SIZE * sizeof(float));

  if (!candidates || !scores) {
    fprintf(stderr, "Error: Memory allocation failed\n");
    if (candidates)
      free(candidates);
    if (scores)
      free(scores);
    free(emb->active_dims);
    free(emb->values);
    emb->num_active = 0;
    emb->active_dims = NULL;
    emb->values = NULL;
    emb->norm = 0.0f;
    return;
  }

  const char *word = vocabulary[word_idx].word;
  float word_length_factor = logf(strlen(word) + 1) / logf(10);

  for (int i = 0; i < EMBEDDING_SIZE; i++) {
    candidates[i] = i;
    scores[i] = 0.0f;

    for (int j = 0; word[j]; j++) {
      if (word[j] >= 'a' && word[j] <= 'z') {
        scores[i] += letter_weights[word[j] - 'a'] * sinf(i * 0.1f + j);
      }
    }

    int category_hash = 0;
    for (int j = 0; vocabulary[word_idx].category[j]; j++) {
      category_hash += vocabulary[word_idx].category[j] * (j + 1);
    }
    scores[i] += sinf(category_hash * 0.001f + i * 0.05f) * word_length_factor;
    scores[i] += ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
  }

  for (int i = 0; i < EMBEDDING_SIZE - 1; i++) {
    for (int j = i + 1; j < EMBEDDING_SIZE; j++) {
      if (scores[j] > scores[i]) {
        float temp_score = scores[i];
        scores[i] = scores[j];
        scores[j] = temp_score;

        int temp_idx = candidates[i];
        candidates[i] = candidates[j];
        candidates[j] = temp_idx;
      }
    }
  }

  for (int i = 0; i < target_active; i++) {
    emb->active_dims[i] = candidates[i];

    float value = 0.0f;
    for (int layer = 0; layer < NUM_SEMANTIC_LAYERS; layer++) {
      int layer_contrib = (word_idx * 17 + candidates[i] * 23 + layer) % 1000;
      value +=
          semantic_weights[layer][candidates[i]] * sinf(layer_contrib * 0.01f);
      emb->semantic_layer[layer] = layer_contrib % EMBEDDING_SIZE;
    }

    emb->values[i] = tanhf(value * vocabulary[word_idx].semantic_weight);
    emb->norm += emb->values[i] * emb->values[i];
  }

  emb->norm = sqrtf(emb->norm);

  if (emb->norm > 1e-8f) {
    for (int i = 0; i < emb->num_active; i++) {
      emb->values[i] /= emb->norm;
    }
    emb->norm = 1.0f;
  }

  free(candidates);
  free(scores);
}

float sparseCosineSimilarity(const SparseEmbedding *a,
                             const SparseEmbedding *b) {
  if (a->norm < 1e-8f || b->norm < 1e-8f)
    return 0.0f;

  float dot_product = 0.0f;
  int i = 0, j = 0;

  while (i < a->num_active && j < b->num_active) {
    if (a->active_dims[i] == b->active_dims[j]) {
      dot_product += a->values[i] * b->values[j];
      i++;
      j++;
    } else if (a->active_dims[i] < b->active_dims[j]) {
      i++;
    } else {
      j++;
    }
  }

  return dot_product;
}

void computeContextHash(char *hash, const char **context_words, int num_words) {
  unsigned int hash_val = 5381;

  for (int i = 0; i < num_words; i++) {
    for (int j = 0; context_words[i][j]; j++) {
      hash_val = ((hash_val << 5) + hash_val) + context_words[i][j];
    }
    hash_val = ((hash_val << 5) + hash_val) + i;
  }

  snprintf(hash, 32, "%u", hash_val);
}

SparseEmbedding *getContextualEmbedding(const char *word, const char **context,
                                        int context_len) {
  int word_idx = -1;
  for (int i = 0; i < vocab_size; i++) {
    if (strcmp(word, vocabulary[i].word) == 0) {
      word_idx = i;
      break;
    }
  }

  if (word_idx == -1)
    return NULL;

  char context_hash[32];
  computeContextHash(context_hash, context, context_len);

  int cache_start = word_idx * 4;
  for (int i = 0; i < 4; i++) {
    if (strcmp(context_cache[cache_start + i].context_hash, context_hash) ==
        0) {
      context_cache[cache_start + i].recency = 1.0f;
      return &context_cache[cache_start + i].embedding;
    }
  }

  int cache_idx = cache_start;
  float min_recency = 1.0f;

  for (int i = 1; i < 4; i++) {
    if (context_cache[cache_start + i].recency < min_recency) {
      min_recency = context_cache[cache_start + i].recency;
      cache_idx = cache_start + i;
    }
  }

  if (context_cache[cache_idx].embedding.active_dims) {
    free(context_cache[cache_idx].embedding.active_dims);
    free(context_cache[cache_idx].embedding.values);
  }

  SparseEmbedding *base = &word_embeddings[word_idx];
  SparseEmbedding *contextual = &context_cache[cache_idx].embedding;

  contextual->num_active = base->num_active;
  contextual->active_dims = malloc(contextual->num_active * sizeof(int));
  contextual->values = malloc(contextual->num_active * sizeof(float));

  memcpy(contextual->active_dims, base->active_dims,
         base->num_active * sizeof(int));
  memcpy(contextual->values, base->values, base->num_active * sizeof(float));
  memcpy(contextual->semantic_layer, base->semantic_layer,
         NUM_SEMANTIC_LAYERS * sizeof(int));

  for (int c = 0; c < context_len && c < CONTEXT_WINDOW; c++) {
    int context_word_idx = -1;
    for (int i = 0; i < vocab_size; i++) {
      if (strcmp(context[c], vocabulary[i].word) == 0) {
        context_word_idx = i;
        break;
      }
    }

    if (context_word_idx != -1) {
      SparseEmbedding *context_emb = &word_embeddings[context_word_idx];
      float context_strength = 0.1f / (c + 1);

      for (int i = 0; i < contextual->num_active; i++) {
        for (int j = 0; j < context_emb->num_active; j++) {
          if (contextual->active_dims[i] == context_emb->active_dims[j]) {
            contextual->values[i] += context_emb->values[j] * context_strength;
            break;
          }
        }
      }
    }
  }

  contextual->norm = 0.0f;
  for (int i = 0; i < contextual->num_active; i++) {
    contextual->norm += contextual->values[i] * contextual->values[i];
  }
  contextual->norm = sqrt(contextual->norm);

  if (contextual->norm > 1e-8f) {
    for (int i = 0; i < contextual->num_active; i++) {
      contextual->values[i] /= contextual->norm;
    }
    contextual->norm = 1.0f;
  }

  strncpy(context_cache[cache_idx].context_hash, context_hash,
          sizeof(context_cache[cache_idx].context_hash) - 1);
  context_cache[cache_idx]
      .context_hash[sizeof(context_cache[cache_idx].context_hash) - 1] = '\0';
  context_cache[cache_idx].recency = 1.0f;

  return contextual;
}

void importPretrainedEmbeddings(const char *embedding_file) {
  if (safe_vocab_size <= 0) {
    fprintf(stderr, "Error: Vocabulary not loaded. Call "
                    "loadVocabularyFromFile first.\n");
    return;
  }

  FILE *file = fopen(embedding_file, "r");
  if (!file) {
    fprintf(stderr, "Error: Could not open embedding file: %s\n",
            embedding_file);
    printf("Falling back to random initialization...\n");

    for (int i = 0; i < safe_vocab_size; i++) {
      for (int j = 0; j < EMBEDDING_SIZE; j++) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        if (u1 < 1e-8f)
          u1 = 1e-8f;
        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
        embeddings[i][j] = z * 0.02f;
      }
    }
    return;
  }

  printf("Loading pre-trained embeddings from %s...\n", embedding_file);

  bool *vocab_found = (bool *)calloc(safe_vocab_size, sizeof(bool));
  if (!vocab_found) {
    fprintf(stderr, "Error: Memory allocation failed\n");
    fclose(file);
    return;
  }

  char *line = (char *)malloc(MAX_LINE_LENGTH);
  if (!line) {
    fprintf(stderr, "Error: Memory allocation failed\n");
    free(vocab_found);
    fclose(file);
    return;
  }

  int file_dim = EMBEDDING_SIZE;
  if (fgets(line, MAX_LINE_LENGTH, file) != NULL) {
    int file_vocab_size;
    if (sscanf(line, "%d %d", &file_vocab_size, &file_dim) == 2) {
      printf("Word2Vec format detected: %d words, %d dimensions\n",
             file_vocab_size, file_dim);

      if (file_dim != EMBEDDING_SIZE) {
        printf("Warning: File embedding size (%d) doesn't match "
               "system size (%d)\n",
               file_dim, EMBEDDING_SIZE);
        printf("Embeddings will be %s\n",
               file_dim > EMBEDDING_SIZE ? "truncated" : "padded with zeros");
      }
    } else {
      rewind(file);
    }
  }

  int loaded_count = 0;
  while (fgets(line, MAX_LINE_LENGTH, file) != NULL) {
    char word[MAX_WORD_LENGTH];
    char *saveptr;
    char *word_token = strtok_r(line, " \t", &saveptr);

    if (!word_token || strlen(word_token) == 0)
      continue;

    strncpy(word, word_token, MAX_WORD_LENGTH - 1);
    word[MAX_WORD_LENGTH - 1] = '\0';

    int vocab_idx = -1;
    for (int i = 0; i < safe_vocab_size; i++) {
      if (strcmp(word, vocabulary[i].word) == 0) {
        vocab_idx = i;
        break;
      }
    }

    if (vocab_idx == -1)
      continue;

    vocab_found[vocab_idx] = true;
    loaded_count++;

    for (int j = 0; j < EMBEDDING_SIZE; j++) {
      char *token = strtok_r(NULL, " \t\n", &saveptr);
      if (token && j < file_dim) {
        embeddings[vocab_idx][j] = atof(token);
      } else {
        embeddings[vocab_idx][j] = 0.0f;
      }
    }
  }

  fclose(file);
  free(line);

  printf("Successfully loaded %d/%d vocabulary words from "
         "pretrained embeddings\n",
         loaded_count, safe_vocab_size);

  for (int i = 0; i < safe_vocab_size; i++) {
    if (!vocab_found[i]) {
      bool found_category_match = false;
      float category_vector[EMBEDDING_SIZE] = {0};
      int category_matches = 0;

      for (int j = 0; j < safe_vocab_size; j++) {
        if (i != j && vocab_found[j] &&
            strcmp(vocabulary[i].category, vocabulary[j].category) == 0) {
          for (int k = 0; k < EMBEDDING_SIZE; k++) {
            category_vector[k] += embeddings[j][k];
          }
          category_matches++;
          found_category_match = true;
        }
      }

      if (found_category_match) {
        for (int k = 0; k < EMBEDDING_SIZE; k++) {
          category_vector[k] /= category_matches;
          float noise = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
          embeddings[i][k] = category_vector[k] + noise;
        }
      } else {
        for (int j = 0; j < EMBEDDING_SIZE; j++) {
          float u1 = (float)rand() / RAND_MAX;
          float u2 = (float)rand() / RAND_MAX;
          if (u1 < 1e-8f)
            u1 = 1e-8f;
          float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
          embeddings[i][j] = z * 0.02f;
        }
      }
    }
  }

  free(vocab_found);

  for (int i = 0; i < safe_vocab_size; ++i) {
    float *emb_i = embeddings[i];
    const char *cat = vocabulary[i].category;

    if (cat) {
      int s, e;
      if (strcmp(cat, "fruit") == 0) {
        clip_range(0, 10, &s, &e);
        for (int j = s; j < e; ++j)
          emb_i[j] += 0.2f;
      } else if (strcmp(cat, "action") == 0) {
        clip_range(10, 20, &s, &e);
        for (int j = s; j < e; ++j)
          emb_i[j] += 0.2f;
      } else if (strcmp(cat, "emotion") == 0) {
        clip_range(20, 30, &s, &e);
        for (int j = s; j < e; ++j)
          emb_i[j] += 0.2f;
      }
    }

    {
      int s, e;
      clip_range(30, 40, &s, &e);
      float lw = vocabulary[i].letter_weight;
      for (int j = s; j < e; ++j)
        emb_i[j] += lw * 0.1f;
    }

    {
      int s, e;
      clip_range(40, 50, &s, &e);
      float sw = vocabulary[i].semantic_weight;
      for (int j = s; j < e; ++j)
        emb_i[j] += sw * 0.1f;
    }

    if (vocabulary[i].connects_to) {
      const char *conn = vocabulary[i].connects_to;
      for (int j = 0; j < safe_vocab_size; ++j) {
        if (j == i)
          continue;
        if (strcmp(conn, vocabulary[j].word) == 0) {
          float *emb_j = embeddings[j];
          int s, e;
          clip_range(50, 60, &s, &e);
          for (int k = s; k < e; ++k) {
            float avg = (emb_i[k] + emb_j[k]) * 0.5f;
            emb_i[k] = emb_i[k] * 0.8f + avg * 0.2f;
            emb_j[k] = emb_j[k] * 0.8f + avg * 0.2f;
          }
          break;
        }
      }
    }
  }

  for (int i = 0; i < safe_vocab_size; i++) {
    float norm = 0.0f;
    for (int j = 0; j < EMBEDDING_SIZE; j++) {
      norm += embeddings[i][j] * embeddings[i][j];
    }
    norm = sqrtf(norm);

    if (norm > 1e-8f) {
      for (int j = 0; j < EMBEDDING_SIZE; j++) {
        embeddings[i][j] /= norm;
      }
    } else {
      for (int j = 0; j < EMBEDDING_SIZE; j++) {
        embeddings[i][j] = 0.0f;
      }
    }
  }

  printf("Embedding initialization completed with custom modifiers "
         "applied\n");
}

void initializeBrainInspiredEmbeddings(const char *pretrained_file) {
  if (safe_vocab_size <= 0) {
    fprintf(stderr, "Error: Vocabulary not loaded\n");
    return;
  }

  printf("Initializing brain-inspired sparse embedding system...\n");

  for (int layer = 0; layer < NUM_SEMANTIC_LAYERS; layer++) {
    for (int i = 0; i < EMBEDDING_SIZE; i++) {
      float freq = 0.01f * (layer + 1);
      semantic_weights[layer][i] = sinf(i * freq) * expf(-layer * 0.1f);
    }
  }

  for (int i = 0; i < safe_vocab_size; i++) {
    initializeSparseEmbedding(&word_embeddings[i], i);
  }

  memset(context_cache, 0, sizeof(context_cache));

  for (int i = 0; i < safe_vocab_size; i++) {
    for (int bucket = 0; bucket < HASH_BUCKETS; bucket++) {
      similarity_hash[bucket][i] = 0.0f;

      if (word_embeddings[i].num_active > 0) {
        for (int j = 0; j < word_embeddings[i].num_active; j++) {
          int dim = word_embeddings[i].active_dims[j];
          if ((dim * 17 + bucket * 23) % 2) {
            similarity_hash[bucket][i] += word_embeddings[i].values[j];
          }
        }
      }
    }
  }

  printf("Brain-inspired embedding system initialized with %d "
         "sparse vectors\n",
         safe_vocab_size);
  printf("Average sparsity: %.1f%% (%.0f active dims per word)\n",
         SPARSE_DENSITY * 100, EMBEDDING_SIZE * SPARSE_DENSITY);
}

void initializeEmbeddings(const char *embedding_file) {
  if (safe_vocab_size <= 0) {
    fprintf(stderr, "Error: Load vocabulary before initializing embeddings\n");
    return;
  }

  importPretrainedEmbeddings(embedding_file);
  initializeBrainInspiredEmbeddings(embedding_file);
}

void cleanupVocabulary() {
  for (int i = 0; i < safe_vocab_size; i++) {
    if (vocabulary[i].connects_to) {
      free((void *)vocabulary[i].connects_to);
      vocabulary[i].connects_to = NULL;
    }
    if (vocabulary[i].description) {
      free((void *)vocabulary[i].description);
      vocabulary[i].description = NULL;
    }
  }
}

void cleanupEmbeddings() {
  for (int i = 0; i < safe_vocab_size; i++) {
    if (word_embeddings[i].active_dims) {
      free(word_embeddings[i].active_dims);
      word_embeddings[i].active_dims = NULL;
    }
    if (word_embeddings[i].values) {
      free(word_embeddings[i].values);
      word_embeddings[i].values = NULL;
    }
  }

  for (int i = 0; i < vocab_size * 4; i++) {
    if (context_cache[i].embedding.active_dims) {
      free(context_cache[i].embedding.active_dims);
      context_cache[i].embedding.active_dims = NULL;
    }
    if (context_cache[i].embedding.values) {
      free(context_cache[i].embedding.values);
      context_cache[i].embedding.values = NULL;
    }
  }
}

float *getWordEmbedding(const char *word, const char **context,
                        int context_len) {
  static float contextual_embedding[EMBEDDING_SIZE];
  int word_index = -1;

  for (int i = 0; i < vocab_size; i++) {
    if (strcmp(word, vocabulary[i].word) == 0) {
      word_index = i;
      break;
    }
  }

  if (word_index == -1) {
    memset(contextual_embedding, 0, EMBEDDING_SIZE * sizeof(float));

    SparseEmbedding *sparse_emb =
        getContextualEmbedding(word, context, context_len);
    if (sparse_emb) {
      for (int i = 0; i < sparse_emb->num_active; i++) {
        contextual_embedding[sparse_emb->active_dims[i]] =
            sparse_emb->values[i];
      }
    } else {
      char *subword_tokens[INPUT_SIZE];
      int num_subwords = 0;
      tokenizeString(word, subword_tokens, &num_subwords);

      if (num_subwords == 0) {
        size_t len = strlen(word);
        for (size_t i = 0; i < len; i++) {
          for (size_t n = 1; n <= 3 && i + n <= len; n++) {
            unsigned int hash = 0;
            char sub[4] = {0};
            // subword n-gram logic
          }
        }
      }
    }
    return contextual_embedding;
  }

  float *dense_emb = embeddings[word_index];
  if (!context || context_len == 0) {
    return dense_emb;
  }

  SparseEmbedding *sparse_emb =
      getContextualEmbedding(word, context, context_len);
  if (sparse_emb) {
    memcpy(contextual_embedding, dense_emb, EMBEDDING_SIZE * sizeof(float));
    for (int i = 0; i < sparse_emb->num_active; i++) {
      int dim = sparse_emb->active_dims[i];
      contextual_embedding[dim] =
          contextual_embedding[dim] * 0.7f + sparse_emb->values[i] * 0.3f;
    }
    return contextual_embedding;
  }

  return dense_emb;
}

void updateEmbeddings(float *feedback, const char *word) {
  for (int i = 0; i < vocab_size; i++) {
    if (strcmp(word, vocabulary[i].word) == 0) {
      for (int j = 0; j < EMBEDDING_SIZE; j++) {
        embeddings[i][j] += feedback[j];
        embeddings[i][j] = fmaxf(0.0f, fminf(1.0f, embeddings[i][j]));
      }
      break;
    }
  }
}

int getTokenIndex(const char *token) {
  for (int i = 0; i < VOCAB_SIZE; i++) {
    if (strcmp(vocabulary[i].word, token) == 0) {
      return i;
    }
  }
  return -1;
}

void createSemanticVector(const char *text, float *vector, int vectorSize,
                          float (*embeddings)[EMBEDDING_SIZE]) {
  const char *delimiters = " ";
  char *textCopy = strdup(text);
  char *token = strtok(textCopy, delimiters);

  for (int i = 0; i < vectorSize; i++) {
    vector[i] = 0.0f;
  }

  int tokenCount = 0;

  while (token != NULL) {
    int tokenIndex = getTokenIndex(token);
    if (tokenIndex != -1) {
      for (int i = 0; i < EMBEDDING_SIZE; i++) {
        vector[i] += embeddings[tokenIndex][i];
      }
      tokenCount++;
    }
    token = strtok(NULL, delimiters);
  }

  if (tokenCount > 0) {
    for (int i = 0; i < EMBEDDING_SIZE; i++) {
      vector[i] /= tokenCount;
    }
  }

  free(textCopy);
}

void storeQuestionAndAnswer(MemorySystem *memorySystem, const char *question,
                            const char *answer, int timestamp) {
  if (memorySystem->size >= memorySystem->capacity) {
    consolidateMemory(memorySystem);
    if (memorySystem->size >= memorySystem->capacity) {
      printf("Warning: Memory system full. Overwriting oldest entry.\n");
    }
  }

  MemoryEntry newEntry;

  float questionVector[MEMORY_VECTOR_SIZE];
  float answerVector[MEMORY_VECTOR_SIZE];

  createSemanticVector(question, questionVector, MEMORY_VECTOR_SIZE,
                       embeddings);
  createSemanticVector(answer, answerVector, MEMORY_VECTOR_SIZE, embeddings);

  for (int i = 0; i < MEMORY_VECTOR_SIZE; i++) {
    newEntry.vector[i] = (questionVector[i] + answerVector[i]) / 2.0f;
  }

  newEntry.importance =
      0.5f + (strlen(question) * 0.01f) + (strlen(answer) * 0.005f);
  newEntry.timestamp = timestamp;

  if (memorySystem->size < memorySystem->capacity) {
    memorySystem->entries[memorySystem->size] = newEntry;
    memorySystem->size++;
  } else {
    int replace_idx = 0;
    float min_importance = memorySystem->entries[0].importance;
    for (int i = 1; i < memorySystem->size; i++) {
      if (memorySystem->entries[i].importance < min_importance) {
        min_importance = memorySystem->entries[i].importance;
        replace_idx = i;
      }
    }
    memorySystem->entries[replace_idx] = newEntry;
  }

  if (newEntry.importance >
      memorySystem->hierarchy.short_term.importance_threshold) {
    if (memorySystem->hierarchy.short_term.size <
        memorySystem->hierarchy.short_term.capacity) {
      memorySystem->hierarchy.short_term
          .entries[memorySystem->hierarchy.short_term.size] = newEntry;
      memorySystem->hierarchy.short_term.size++;
    } else {
      int st_replace_idx = 0;
      float st_min_importance =
          memorySystem->hierarchy.short_term.entries[0].importance;
      for (int i = 1; i < memorySystem->hierarchy.short_term.size; i++) {
        if (memorySystem->hierarchy.short_term.entries[i].importance <
            st_min_importance) {
          st_min_importance =
              memorySystem->hierarchy.short_term.entries[i].importance;
          st_replace_idx = i;
        }
      }
      if (newEntry.importance > st_min_importance) {
        memorySystem->hierarchy.short_term.entries[st_replace_idx] = newEntry;
      }
    }
  }
}

void updateContextAnswer(GlobalContextManager *contextManager,
                         const char *question, const char *answer) {
  ContextNode *currentNode = contextManager->root;
  char contextName[64] = "QA_Interaction";
  bool found = false;

  for (uint32_t i = 0; i < currentNode->num_children; i++) {
    if (strcmp(currentNode->children[i]->name, contextName) == 0) {
      currentNode = currentNode->children[i];
      found = true;
      break;
    }
  }

  if (!found) {
    if (currentNode->num_children < currentNode->max_children) {
      ContextNode *newNode = malloc(sizeof(ContextNode));
      newNode->name = strdup(contextName);
      newNode->importance = 0.7f;
      newNode->vector_size = contextManager->vector_size;
      newNode->state_vector = malloc(sizeof(float) * newNode->vector_size);

      float contextNameVector[MEMORY_VECTOR_SIZE] = {0.0f};
      for (uint32_t i = 0; i < MEMORY_VECTOR_SIZE; i++) {
        newNode->state_vector[i] = contextNameVector[i];
      }

      newNode->children = NULL;
      newNode->num_children = 0;
      newNode->max_children = contextManager->max_children_per_node;
      newNode->parent = currentNode;
      newNode->temporal_relevance = 1.0f;
      newNode->last_updated = time(NULL);

      currentNode->children =
          realloc(currentNode->children,
                  sizeof(ContextNode *) * (currentNode->num_children + 1));
      currentNode->children[currentNode->num_children] = newNode;
      currentNode->num_children++;
      contextManager->total_nodes++;
      currentNode = newNode;
    }
  }

  if (currentNode != contextManager->root) {
    float questionVector[MEMORY_VECTOR_SIZE];
    float answerVector[MEMORY_VECTOR_SIZE];

    createSemanticVector(question, questionVector, MEMORY_VECTOR_SIZE,
                         embeddings);
    createSemanticVector(answer, answerVector, MEMORY_VECTOR_SIZE, embeddings);

    float combinedVector[MEMORY_VECTOR_SIZE];
    for (uint32_t i = 0; i < MEMORY_VECTOR_SIZE; i++) {
      combinedVector[i] = (questionVector[i] + answerVector[i]) / 2.0f;
    }

    for (uint32_t i = 0; i < currentNode->vector_size; i++) {
      float semanticInfluence =
          (i < MEMORY_VECTOR_SIZE) ? combinedVector[i] : 0.0f;
      currentNode->state_vector[i] =
          (currentNode->state_vector[i] * (1.0f - contextManager->decay_rate)) +
          (semanticInfluence * contextManager->decay_rate);
    }

    currentNode->last_updated = time(NULL);
    currentNode->temporal_relevance = 1.0f;

    for (uint32_t i = 0; i < contextManager->vector_size; i++) {
      contextManager->global_context_vector[i] =
          (contextManager->global_context_vector[i] *
           (1.0f - contextManager->decay_rate)) +
          (currentNode->state_vector[i] * currentNode->importance *
           contextManager->decay_rate);
    }
  }
}

static unsigned int fnv1a_hash(const char *str) {
  unsigned int hash = 2166136261u;
  while (*str) {
    hash ^= (unsigned char)*str++;
    hash *= 16777619u;
  }
  return hash;
}

static void simple_stem(char *word) {
  int len = strlen(word);
  if (len < 4)
    return;

  for (int i = 0; word[i]; i++) {
    word[i] = tolower(word[i]);
  }

  if (len > 4) {
    if (strcmp(word + len - 3, "ing") == 0) {
      word[len - 3] = '\0';
    } else if (strcmp(word + len - 2, "ed") == 0) {
      word[len - 2] = '\0';
    } else if (strcmp(word + len - 2, "er") == 0) {
      word[len - 2] = '\0';
    } else if (strcmp(word + len - 1, "s") == 0 && word[len - 2] != 's') {
      word[len - 1] = '\0';
    }
  }
}

static int is_stop_word(const char *word) {
  static const char *stop_words[] = {
      "the",   "a",      "an",  "and",   "or",  "but",  "in",
      "on",    "at",     "to",  "for",   "of",  "with", "by",
      "is",    "are",    "was", "were",  "be",  "been", "have",
      "has",   "had",    "do",  "does",  "did", "will", "would",
      "could", "should", "may", "might", "can", "this", "that",
      "these", "those",  "i",   "you",   "he",  "she",  "it",
      "we",    "they",   "me",  "him",   "her", "us",   "them"};

  int num_stop_words = sizeof(stop_words) / sizeof(stop_words[0]);
  for (int i = 0; i < num_stop_words; i++) {
    if (strcmp(word, stop_words[i]) == 0) {
      return 1;
    }
  }
  return 0;
}

static int tokenize_text(const char *text, char tokens[][MAX_TOKEN_LENGTH]) {
  char text_copy[MAX_TEXT_LENGTH];
  strncpy(text_copy, text, MAX_TEXT_LENGTH - 1);
  text_copy[MAX_TEXT_LENGTH - 1] = '\0';

  int num_tokens = 0;
  char *token = strtok(text_copy, " \t\n\r.,!?;:()[]{}\"'");

  while (token != NULL && num_tokens < MAX_TOKENS) {
    int len = strlen(token);
    if (len >= 2 && len < MAX_TOKEN_LENGTH) {
      char processed_token[MAX_TOKEN_LENGTH];
      strncpy(processed_token, token, MAX_TOKEN_LENGTH - 1);
      processed_token[MAX_TOKEN_LENGTH - 1] = '\0';

      for (int i = 0; processed_token[i]; i++) {
        processed_token[i] = tolower(processed_token[i]);
      }

      if (!is_stop_word(processed_token)) {
        simple_stem(processed_token);
        strncpy(tokens[num_tokens], processed_token, MAX_TOKEN_LENGTH - 1);
        tokens[num_tokens][MAX_TOKEN_LENGTH - 1] = '\0';
        num_tokens++;
      }
    }
    token = strtok(NULL, " \t\n\r.,!?;:()[]{}\"'");
  }

  return num_tokens;
}

static void add_ngrams_to_vector(float *memory_vector,
                                 char tokens[][MAX_TOKEN_LENGTH],
                                 int num_tokens, int n) {
  char ngram[MAX_TOKEN_LENGTH * NGRAM_SIZE];

  for (int i = 0; i <= num_tokens - n; i++) {
    strcpy(ngram, tokens[i]);
    for (int j = 1; j < n; j++) {
      strcat(ngram, "_");
      strcat(ngram, tokens[i + j]);
    }

    unsigned int hash = fnv1a_hash(ngram);
    int index = hash % MEMORY_VECTOR_SIZE;
    memory_vector[index] += 1.0f / (float)n;
  }
}

static void apply_tf_weighting(float *memory_vector,
                               char tokens[][MAX_TOKEN_LENGTH],
                               int num_tokens) {
  float tf_counts[MEMORY_VECTOR_SIZE] = {0};

  for (int i = 0; i < num_tokens; i++) {
    unsigned int hash = fnv1a_hash(tokens[i]);
    int index = hash % MEMORY_VECTOR_SIZE;
    tf_counts[index] += 1.0f;
  }

  for (int i = 0; i < MEMORY_VECTOR_SIZE; i++) {
    if (tf_counts[i] > 0) {
      memory_vector[i] *= (1.0f + logf(tf_counts[i]));
    }
  }
}

void computeMemoryVectorFromText(float *memory_vector, const char *question,
                                 const char *answer) {
  memset(memory_vector, 0, MEMORY_VECTOR_SIZE * sizeof(float));

  char combined_text[MAX_TEXT_LENGTH];
  snprintf(combined_text, sizeof(combined_text), "%s %s %s", question, question,
           answer);

  char tokens[MAX_TOKENS][MAX_TOKEN_LENGTH];
  int num_tokens = tokenize_text(combined_text, tokens);

  if (num_tokens == 0)
    return;

  for (int i = 0; i < num_tokens; i++) {
    unsigned int hash = fnv1a_hash(tokens[i]);
    int index = hash % MEMORY_VECTOR_SIZE;
    memory_vector[index] += 1.0f;
  }

  if (num_tokens > 1) {
    add_ngrams_to_vector(memory_vector, tokens, num_tokens, 2);
  }

  if (num_tokens > 2) {
    add_ngrams_to_vector(memory_vector, tokens, num_tokens, 3);
  }

  apply_tf_weighting(memory_vector, tokens, num_tokens);

  for (int i = 0; i < num_tokens; i++) {
    unsigned int hash = fnv1a_hash(tokens[i]);
    int index = hash % MEMORY_VECTOR_SIZE;
    float position_weight = 1.0f + (1.0f / (1.0f + (float)i * 0.1f));
    memory_vector[index] *= position_weight;
  }

  float norm = 0.0f;
  for (int i = 0; i < MEMORY_VECTOR_SIZE; i++) {
    norm += memory_vector[i] * memory_vector[i];
  }

  norm = sqrtf(norm);
  if (norm > 1e-8f) {
    for (int i = 0; i < MEMORY_VECTOR_SIZE; i++) {
      memory_vector[i] /= norm;
    }
  }
}

float computeImportanceFromText(const char *question, const char *answer) {
  float importance = 0.0f;

  int question_length = strlen(question);
  int answer_length = strlen(answer);
  importance += 0.1f * (question_length + answer_length);

  const char *keywords[] = {"error", "goal", "priority", "critical",
                            "important"};
  int num_keywords = sizeof(keywords) / sizeof(keywords[0]);
  for (int i = 0; i < num_keywords; i++) {
    if (strstr(question, keywords[i])) {
      importance += 5.0f;
    }
    if (strstr(answer, keywords[i])) {
      importance += 5.0f;
    }
  }

  if (strstr(answer, "Pattern Recognition")) {
    importance += 8.0f;
  }
  if (strstr(answer, "Numerical Computation")) {
    importance += 7.5f;
  }
  if (strstr(answer, "Sequence Learning")) {
    importance += 9.0f;
  }
  if (strstr(answer, "Classification")) {
    importance += 7.0f;
  }
  if (strstr(answer, "Prediction")) {
    importance += 10.0f;
  }
  if (strstr(answer, "Optimization")) {
    importance += 8.5f;
  }
  if (strstr(answer, "Error Correction")) {
    importance += 9.5f;
  }
  if (strstr(answer, "Memory Consolidation")) {
    importance += 6.0f;
  }

  importance = fminf(importance, 100.0f);
  importance = fmaxf(importance, 0.0f);

  return importance;
}

void addQuestionAndAnswerToMemory(
    MemorySystem *memorySystem, WorkingMemorySystem *workingMemory,
    const char *question, const char *answer,
    float feature_projection_matrix[FEATURE_VECTOR_SIZE][MEMORY_VECTOR_SIZE]) {
  MemoryEntry entry;
  entry.timestamp = getCurrentTime();
  entry.importance = computeImportanceFromText(question, answer);

  computeMemoryVectorFromText(entry.vector, question, answer);

  if (entry.importance > workingMemory->focus.attention_threshold) {
    if (workingMemory->focus.size < workingMemory->focus.capacity) {
      WorkingMemoryEntry enhanced;
      enhanced.features = malloc(FEATURE_VECTOR_SIZE * sizeof(float));
      extractSemanticFeatures(entry.vector, enhanced.features,
                              feature_projection_matrix);
      enhanced.context_vector = malloc(CONTEXT_VECTOR_SIZE * sizeof(float));
      memcpy(enhanced.context_vector, workingMemory->global_context,
             CONTEXT_VECTOR_SIZE * sizeof(float));
      workingMemory->focus.entries[workingMemory->focus.size++] = enhanced;
      updateSemanticClusters(workingMemory, &enhanced);
    }
  } else {
    if (workingMemory->active.size < workingMemory->active.capacity) {
      WorkingMemoryEntry enhanced;
      enhanced.features = malloc(FEATURE_VECTOR_SIZE * sizeof(float));
      extractSemanticFeatures(entry.vector, enhanced.features,
                              feature_projection_matrix);
      enhanced.context_vector = malloc(CONTEXT_VECTOR_SIZE * sizeof(float));
      memcpy(enhanced.context_vector, workingMemory->global_context,
             CONTEXT_VECTOR_SIZE * sizeof(float));
      workingMemory->active.entries[workingMemory->active.size++] = enhanced;
      updateSemanticClusters(workingMemory, &enhanced);
    }
  }

  updateContext(workingMemory);

  if (entry.importance >=
      memorySystem->hierarchy.long_term.importance_threshold) {
    if (memorySystem->hierarchy.long_term.size <
        memorySystem->hierarchy.long_term.capacity) {
      memorySystem->hierarchy.long_term
          .entries[memorySystem->hierarchy.long_term.size++] = entry;
    } else {
      unsigned int replace_count;
      int *least_important = findLeastImportantMemory(
          memorySystem->hierarchy.long_term.entries,
          memorySystem->hierarchy.long_term.size, 10, &replace_count);

      if (least_important && replace_count > 0) {
        float worst_importance =
            memorySystem->hierarchy.long_term.entries[least_important[0]]
                .importance;
        if (entry.importance > worst_importance * 1.2f) {
          memorySystem->hierarchy.long_term.entries[least_important[0]] = entry;
        }
        free(least_important);
      }
    }
  } else if (entry.importance >=
             memorySystem->hierarchy.medium_term.importance_threshold) {
    if (memorySystem->hierarchy.medium_term.size <
        memorySystem->hierarchy.medium_term.capacity) {
      memorySystem->hierarchy.medium_term
          .entries[memorySystem->hierarchy.medium_term.size++] = entry;
    } else {
      unsigned int replace_count;
      int *least_important = findLeastImportantMemory(
          memorySystem->hierarchy.medium_term.entries,
          memorySystem->hierarchy.medium_term.size, 7, &replace_count);

      if (least_important && replace_count > 0) {
        int best_replacement = -1;
        float best_score = -1.0f;

        for (unsigned int i = 0; i < replace_count; i++) {
          int idx = least_important[i];
          MemoryEntry *candidate =
              &memorySystem->hierarchy.medium_term.entries[idx];
          unsigned int age = entry.timestamp - candidate->timestamp;
          float score =
              (1.0f / (candidate->importance + 0.1f)) + (age * 0.001f);
          if (score > best_score) {
            best_score = score;
            best_replacement = idx;
          }
        }

        if (best_replacement >= 0) {
          memorySystem->hierarchy.medium_term.entries[best_replacement] = entry;
        }
        free(least_important);
      } else {
        consolidateToHigherLevel(memorySystem);
      }
    }
  } else {
    if (memorySystem->hierarchy.short_term.size <
        memorySystem->hierarchy.short_term.capacity) {
      memorySystem->hierarchy.short_term
          .entries[memorySystem->hierarchy.short_term.size++] = entry;
    } else {
      unsigned int replace_count;
      int *least_important = findLeastImportantMemory(
          memorySystem->hierarchy.short_term.entries,
          memorySystem->hierarchy.short_term.size,
          memorySystem->hierarchy.short_term.size / 3, &replace_count);

      if (least_important && replace_count > 0) {
        int oldest_idx = least_important[0];
        unsigned int oldest_time =
            memorySystem->hierarchy.short_term.entries[oldest_idx].timestamp;

        for (unsigned int i = 1; i < replace_count; i++) {
          int idx = least_important[i];
          if (memorySystem->hierarchy.short_term.entries[idx].timestamp <
              oldest_time) {
            oldest_time =
                memorySystem->hierarchy.short_term.entries[idx].timestamp;
            oldest_idx = idx;
          }
        }

        memorySystem->hierarchy.short_term.entries[oldest_idx] = entry;
        free(least_important);
      } else {
        consolidateToMediumTerm(memorySystem);
      }
    }
  }

  memorySystem->entries[memorySystem->head] = entry;
  memorySystem->head = (memorySystem->head + 1) % memorySystem->capacity;
  if (memorySystem->size < memorySystem->capacity) {
    memorySystem->size++;
  }
}

void getEmotionName(int emotion_id, char *name) {
  static const char *emotion_names[] = {"love", "hate", "joy", "fear"};

  if (emotion_id >= 0 && emotion_id < MAX_EMOTION_TYPES &&
      emotion_id < sizeof(emotion_names) / sizeof(emotion_names[0])) {
    strcpy(name, emotion_names[emotion_id]);
  } else {
    strcpy(name, "unknown");
  }
}

void expandMemoryCapacity(MemorySystem *memorySystem) {
  unsigned int new_capacity = memorySystem->capacity * 1.5;
  MemoryEntry *new_entries =
      (MemoryEntry *)malloc(new_capacity * sizeof(MemoryEntry));
  if (!new_entries) {
    fprintf(stderr, "Failed to expand memory capacity.\n");
    return;
  }

  for (int i = 0; i < memorySystem->size; i++) {
    new_entries[i] =
        memorySystem
            ->entries[(memorySystem->head + i) % memorySystem->capacity];
  }

  free(memorySystem->entries);
  memorySystem->entries = new_entries;
  memorySystem->capacity = new_capacity;
  memorySystem->head = 0;
}

float calculatePerformanceStability(float *performance_history,
                                    int history_length) {
  if (history_length <= 1) {
    return 1.0f;
  }

  float mean = 0.0f;
  for (int i = 0; i < history_length; i++) {
    mean += performance_history[i];
  }
  mean /= history_length;

  float variance = 0.0f;
  for (int i = 0; i < history_length; i++) {
    float diff = performance_history[i] - mean;
    variance += diff * diff;
  }
  variance /= history_length;
  float std_dev = sqrtf(variance);

  float cv = (mean != 0.0f) ? std_dev / mean : std_dev;

  int direction_changes = 0;
  int prev_direction = 0;

  for (int i = 1; i < history_length; i++) {
    int current_direction = 0;
    if (performance_history[i] > performance_history[i - 1]) {
      current_direction = 1;
    } else if (performance_history[i] < performance_history[i - 1]) {
      current_direction = -1;
    }

    if (prev_direction != 0 && current_direction != 0 &&
        current_direction != prev_direction) {
      direction_changes++;
    }

    if (current_direction != 0) {
      prev_direction = current_direction;
    }
  }

  float max_possible_changes = history_length - 2;
  float direction_stability =
      (max_possible_changes > 0)
          ? 1.0f - (direction_changes / max_possible_changes)
          : 1.0f;

  float recent_stability = 0.0f;
  int recent_window = history_length / 3;
  if (recent_window > 1) {
    float recent_variance = 0.0f;
    float recent_mean = 0.0f;

    for (int i = history_length - recent_window; i < history_length; i++) {
      recent_mean += performance_history[i];
    }
    recent_mean /= recent_window;

    for (int i = history_length - recent_window; i < history_length; i++) {
      float diff = performance_history[i] - recent_mean;
      recent_variance += diff * diff;
    }
    recent_variance /= recent_window;

    float recent_std_dev = sqrtf(recent_variance);
    float recent_cv =
        (recent_mean != 0.0f) ? recent_std_dev / recent_mean : recent_std_dev;

    recent_stability = (recent_cv <= 0.5f) ? 1.0f - (recent_cv / 0.5f) : 0.0f;
  } else {
    recent_stability = 1.0f;
  }

  float cv_stability = (cv <= 0.5f) ? 1.0f - (cv / 0.5f) : 0.0f;

  float overall_stability = (0.4f * cv_stability) +
                            (0.3f * direction_stability) +
                            (0.3f * recent_stability);

  overall_stability = fmaxf(0.0f, fminf(1.0f, overall_stability));

  return overall_stability;
}

void adjustBehaviorBasedOnAnswers(
    Neuron *neurons, float *input_tensor, MemorySystem *memorySystem,
    float *learning_rate, float *input_noise_scale, float *weight_noise_scale,
    NetworkStateSnapshot *stateSnapshot, GlobalContextManager *contextManager,
    IntrinsicMotivation *motivation, GoalSystem *goalSystem,
    WorkingMemorySystem *workingMemory, SelfIdentitySystem *identitySystem,
    MetacognitionMetrics *metacognition, DynamicParameters *dynamicParams,
    MetaLearningState *metaLearning, EmotionalSystem *emotionalSystem,
    ImaginationSystem *imaginationSystem, SocialSystem *socialSystem) {
  float error_rate = computeErrorRate(neurons, input_tensor);
  if (error_rate > 0.5) {
    printf("Error rate is high. Increasing learning rate.\n");
    *learning_rate *= 1.1f;

    metaLearning->learning_efficiency *= 0.9f;
    printf("Decreased learning efficiency to %.2f due to high error rate.\n",
           metaLearning->learning_efficiency);
  } else if (error_rate < 0.2) {
    metaLearning->learning_efficiency =
        fmin(1.0f, metaLearning->learning_efficiency * 1.05f);
    printf("Increased learning efficiency to %.2f due to low error rate.\n",
           metaLearning->learning_efficiency);
  }

  if (error_rate > 0.5) {
    printf("Error rate is high (%.2f). Increasing input noise.\n", error_rate);
    *input_noise_scale = fmin(1.0f, *input_noise_scale + 0.1f);

    motivation->exploration_rate =
        fmin(1.0f, motivation->exploration_rate + 0.05f);
    printf("Increased exploration rate to %.2f\n",
           motivation->exploration_rate);
  } else if (error_rate < 0.2) {
    printf("Error rate is low (%.2f). Decreasing input noise.\n", error_rate);
    *input_noise_scale = fmax(0.0f, *input_noise_scale - 0.1f);

    motivation->exploration_rate =
        fmax(0.1f, motivation->exploration_rate - 0.05f);
    printf("Decreased exploration rate to %.2f\n",
           motivation->exploration_rate);
  }

  if (error_rate > 0.5) {
    printf("Error rate is high (%.2f). Increasing weight noise.\n", error_rate);
    *weight_noise_scale = fmin(1.0f, *weight_noise_scale + 0.1f);

    dynamicParams->plasticity = fmin(1.0f, dynamicParams->plasticity + 0.1f);
    printf("Increased plasticity to %.2f\n", dynamicParams->plasticity);
  } else if (error_rate < 0.2) {
    printf("Error rate is low (%.2f). Decreasing weight noise.\n", error_rate);
    *weight_noise_scale = fmax(0.0f, *weight_noise_scale - 0.1f);

    dynamicParams->plasticity = fmax(0.1f, dynamicParams->plasticity - 0.05f);
    printf("Decreased plasticity to %.2f\n", dynamicParams->plasticity);
  }

  float usage_ratio = (float)memorySystem->size / memorySystem->capacity;

  if (usage_ratio >= 0.8f && usage_ratio < 0.95f) {
    printf("Memory usage is high (%.2f%%). Consolidating.\n",
           usage_ratio * 100.0f);
    consolidateMemory(memorySystem);

    memorySystem->hierarchy.consolidation_threshold *= 0.9f;
    printf("Lowered consolidation threshold to %.2f\n",
           memorySystem->hierarchy.consolidation_threshold);

  } else if (usage_ratio >= 0.95f) {
    printf("Memory usage is critical (%.2f%%). Expanding.\n",
           usage_ratio * 100.0f);
    expandMemoryCapacity(memorySystem);

    memorySystem->hierarchy.consolidation_threshold = 0.5f;
    printf("Reset consolidation threshold to %.2f\n",
           memorySystem->hierarchy.consolidation_threshold);
  }

  if (metacognition->cognitive_load > 0.7f) {
    workingMemory->focus.attention_threshold += 0.05f;
    printf("High cognitive load (%.2f). Increased threshold to %.2f\n",
           metacognition->cognitive_load,
           workingMemory->focus.attention_threshold);
  } else if (metacognition->cognitive_load < 0.3f) {
    workingMemory->focus.attention_threshold =
        fmax(0.1f, workingMemory->focus.attention_threshold - 0.05f);
    printf("Low cognitive load (%.2f). Decreased threshold to %.2f\n",
           metacognition->cognitive_load,
           workingMemory->focus.attention_threshold);
  }

  if (metacognition->error_awareness > 0.6f) {
    contextManager->decay_rate =
        fmin(0.99f, contextManager->decay_rate + 0.05f);
    printf("High error awareness (%.2f). Increased decay to %.2f\n",
           metacognition->error_awareness, contextManager->decay_rate);
  } else if (metacognition->error_awareness < 0.3f) {
    contextManager->decay_rate = fmax(0.2f, contextManager->decay_rate - 0.05f);
    printf("Low error awareness (%.2f). Decreased decay to %.2f\n",
           metacognition->error_awareness, contextManager->decay_rate);
  }

  if (error_rate < 0.2f && metacognition->confidence_level > 0.7f) {
    int highest_priority_idx = -1;
    float highest_priority = -1.0f;

    for (int i = 0; i < goalSystem->num_goals; i++) {
      if (!goalSystem->goals[i].achieved &&
          goalSystem->goals[i].priority > highest_priority) {
        highest_priority = goalSystem->goals[i].priority;
        highest_priority_idx = i;
      }
    }

    if (highest_priority_idx >= 0) {
      goalSystem->goals[highest_priority_idx].reward_value *= 1.1f;
      printf("Increased reward for goal '%s' to %.2f\n",
             goalSystem->goals[highest_priority_idx].description,
             goalSystem->goals[highest_priority_idx].reward_value);
    }
  }

  int dominant_emotion = 0;
  float max_intensity = 0.0f;
  for (int j = 0; j < MAX_EMOTION_TYPES; j++) {
    if (emotionalSystem->emotions[j].intensity > max_intensity) {
      max_intensity = emotionalSystem->emotions[j].intensity;
      dominant_emotion = j;
    }
  }

  if (max_intensity > 0.7f) {
    emotionalSystem->cognitive_impact =
        fmin(1.0f, emotionalSystem->cognitive_impact + 0.05f);
    printf("High emotional intensity (%.2f). Increased impact to %.2f\n",
           max_intensity, emotionalSystem->cognitive_impact);

    if (emotionalSystem->emotional_regulation < 0.5f) {
      emotionalSystem->emotional_regulation += 0.03f;
      printf("Increased regulation to %.2f\n",
             emotionalSystem->emotional_regulation);
    }
  } else if (max_intensity < 0.3f) {
    emotionalSystem->cognitive_impact =
        fmax(0.1f, emotionalSystem->cognitive_impact - 0.03f);
    printf("Low emotional intensity (%.2f). Decreased impact to %.2f\n",
           max_intensity, emotionalSystem->cognitive_impact);
  }

  if (error_rate > 0.5f && emotionalSystem->emotional_regulation < 0.7f) {
    emotionalSystem->emotional_regulation += 0.05f;
    printf("High error rate. Increased regulation to %.2f\n",
           emotionalSystem->emotional_regulation);
  }

  int memory_idx = emotionalSystem->memory_index;
  memory_idx = (memory_idx + 1) % 10;
  for (int i = 0; i < MAX_EMOTION_TYPES; i++) {
    emotionalSystem->emotional_memory[i][memory_idx] =
        emotionalSystem->emotions[i].intensity;
  }
  emotionalSystem->memory_index = memory_idx;

  if (metacognition->cognitive_load < 0.4f) {
    imaginationSystem->creativity_factor =
        fmin(1.0f, imaginationSystem->creativity_factor + 0.05f);
    printf("Low cognitive load. Increased creativity to %.2f\n",
           imaginationSystem->creativity_factor);
  } else if (metacognition->cognitive_load > 0.7f) {
    imaginationSystem->creativity_factor =
        fmax(0.2f, imaginationSystem->creativity_factor - 0.05f);
    printf("High cognitive load. Decreased creativity to %.2f\n",
           imaginationSystem->creativity_factor);

    imaginationSystem->coherence_threshold += 0.03f;
    printf("Increased coherence threshold to %.2f\n",
           imaginationSystem->coherence_threshold);
  }

  if (motivation->exploration_rate > 0.6f) {
    imaginationSystem->novelty_weight =
        fmin(1.0f, imaginationSystem->novelty_weight + 0.05f);
    printf("High exploration. Increased novelty weight to %.2f\n",
           imaginationSystem->novelty_weight);
  } else if (motivation->exploration_rate < 0.3f) {
    imaginationSystem->novelty_weight =
        fmax(0.1f, imaginationSystem->novelty_weight - 0.03f);
    printf("Low exploration. Decreased novelty weight to %.2f\n",
           imaginationSystem->novelty_weight);
  }

  if (error_rate > 0.6f && metacognition->confidence_level < 0.4f &&
      !imaginationSystem->active) {
    imaginationSystem->active = true;
    printf("Activating imagination system\n");

    strcpy(imaginationSystem->current_scenario_name, "problem_solving");
    imaginationSystem->current_scenario = 0;
    imaginationSystem->scenarios[0].num_outcomes = 0;
    imaginationSystem->scenarios[0].divergence_factor = 0.7f;
  }

  if (imaginationSystem->active && error_rate < 0.2f &&
      metacognition->confidence_level > 0.7f) {
    imaginationSystem->active = false;
    printf("Deactivating imagination system\n");
    imaginationSystem->total_scenarios_generated++;
  }

  if (emotionalSystem->emotional_regulation > 0.6f) {
    socialSystem->empathy_level =
        fmin(1.0f, socialSystem->empathy_level + 0.03f);
    printf("Good regulation. Increased empathy to %.2f\n",
           socialSystem->empathy_level);
  } else if (emotionalSystem->emotional_regulation < 0.3f) {
    socialSystem->empathy_level =
        fmax(0.3f, socialSystem->empathy_level - 0.03f);
    printf("Poor regulation. Decreased empathy to %.2f\n",
           socialSystem->empathy_level);
  }

  if (metaLearning->learning_efficiency > 0.7f) {
    socialSystem->learning_rate =
        fmin(0.5f, socialSystem->learning_rate * 1.05f);
    printf("High efficiency. Increased social LR to %.3f\n",
           socialSystem->learning_rate);
  } else if (metaLearning->learning_efficiency < 0.4f) {
    socialSystem->learning_rate =
        fmax(0.05f, socialSystem->learning_rate * 0.95f);
    printf("Low efficiency. Decreased social LR to %.3f\n",
           socialSystem->learning_rate);
  }

  if (identitySystem->consistency_score > 0.7f) {
    socialSystem->negotiation_skill =
        fmin(1.0f, socialSystem->negotiation_skill + 0.02f);
    printf("Strong identity. Increased negotiation to %.2f\n",
           socialSystem->negotiation_skill);
  }

  float performance_stability = calculatePerformanceStability(
      metacognition->performance_history, HISTORY_LENGTH);

  if (performance_stability > 0.7f) {
    socialSystem->behavior_prediction_accuracy =
        fmin(1.0f, socialSystem->behavior_prediction_accuracy + 0.02f);
    printf("Stable performance. Increased pred accuracy to %.2f\n",
           socialSystem->behavior_prediction_accuracy);
  } else if (performance_stability < 0.3f) {
    socialSystem->behavior_prediction_accuracy =
        fmax(0.3f, socialSystem->behavior_prediction_accuracy - 0.02f);
    printf("Unstable performance. Decreased pred accuracy to %.2f\n",
           socialSystem->behavior_prediction_accuracy);
  }

  float performance_stability_with_emotion =
      performance_stability * (1.0f - 0.3f * emotionalSystem->cognitive_impact);

  if (performance_stability_with_emotion > 0.8f) {
    identitySystem->adaptation_rate *= 0.95f;
    printf("Stable with emotions (%.2f). Decreased identity rate to %.4f\n",
           performance_stability_with_emotion, identitySystem->adaptation_rate);
  } else if (performance_stability_with_emotion < 0.3f) {
    identitySystem->adaptation_rate =
        fmin(0.2f, identitySystem->adaptation_rate * 1.1f);
    printf("Unstable (%.2f). Increased identity rate to %.4f\n",
           performance_stability_with_emotion, identitySystem->adaptation_rate);
  }
}

void askQuestion(
    int question_id, Neuron *neurons, float *input_tensor,
    MemorySystem *memorySystem, float *learning_rate,
    NetworkStateSnapshot *stateSnapshot, GlobalContextManager *contextManager,
    IntrinsicMotivation *motivation, GoalSystem *goalSystem,
    WorkingMemorySystem *workingMemory, SelfIdentitySystem *identitySystem,
    MetacognitionMetrics *metacognition, KnowledgeFilter *filter,
    EmotionalSystem *emotionalSystem, ImaginationSystem *imaginationSystem,
    SocialSystem *socialSystem,
    float feature_projection_matrix[FEATURE_VECTOR_SIZE][MEMORY_VECTOR_SIZE]) {
  if (question_id < 0 || question_id >= num_questions) {
    printf("Invalid question ID\n");
    return;
  }

  InternalQuestion *question = &question_table[question_id];
  char fullQuestionStr[1024] = "";
  char fullAnswerStr[1024] = "";

  for (int i = 0; i < question->num_symbols; i++) {
    int symbol_id = question->symbol_ids[i];
    if (symbol_id < 0 || symbol_id >= num_symbols) {
      printf("Invalid symbol ID\n");
      continue;
    }

    InternalSymbol *symbol = &symbol_table[symbol_id];
    printf("Question: %s\n", symbol->description);

    strcat(fullQuestionStr, symbol->description);
    strcat(fullQuestionStr, " ");

    char answerBuffer[256] = "";

    if (symbol_id == 0) {
      if (filter->num_categories > 0) {
        KnowledgeCategory *last_category = &filter->categories[0];
        for (uint32_t i = 1; i < filter->num_categories; i++) {
          if (filter->categories[i].last_accessed >
              last_category->last_accessed) {
            last_category = &filter->categories[i];
          }
        }
        snprintf(answerBuffer, sizeof(last_category->name), "%s",
                 last_category->name);
      }
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 1) {
      float error_rate = computeErrorRate(neurons, input_tensor);
      sprintf(answerBuffer, "Current error rate is %.2f", error_rate);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 2) {
      sprintf(answerBuffer, "Current learning rate is %.4f", *learning_rate);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 3) {
      sprintf(answerBuffer, "Current memory usage is %u/%u", memorySystem->size,
              memorySystem->capacity);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 4) {
      sprintf(answerBuffer, "Short-term memory has %u/%u entries",
              memorySystem->hierarchy.short_term.size,
              memorySystem->hierarchy.short_term.capacity);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 5) {
      sprintf(answerBuffer, "Long-term memory has %u/%u entries",
              memorySystem->hierarchy.long_term.size,
              memorySystem->hierarchy.long_term.capacity);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 6) {
      sprintf(answerBuffer, "Current network step is %d", stateSnapshot->step);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 7) {
      sprintf(answerBuffer, "Global context has %u total nodes with decay %.4f",
              contextManager->total_nodes, contextManager->decay_rate);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 8) {
      float avg_prediction_error = 0.0f;
      for (int j = 0; j < MAX_NEURONS; j++) {
        avg_prediction_error += predictive_params[j].prediction_error;
      }
      avg_prediction_error /= MAX_NEURONS;
      sprintf(answerBuffer, "Average prediction error across neurons is %.4f",
              avg_prediction_error);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 9) {
      sprintf(answerBuffer,
              "Working memory focus has %u/%u entries "
              "with attention threshold %.4f",
              workingMemory->focus.size, workingMemory->focus.capacity,
              workingMemory->focus.attention_threshold);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 10) {
      sprintf(answerBuffer,
              "Current curiosity drive is %.2f "
              "with exploration rate %.2f",
              motivation->curiosity_drive, motivation->exploration_rate);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 11) {
      int active_goals = 0;
      for (int j = 0; j < goalSystem->num_goals; j++) {
        if (!goalSystem->goals[j].achieved) {
          active_goals++;
        }
      }
      sprintf(answerBuffer, "System has %d active goals out of %d total goals",
              active_goals, goalSystem->num_goals);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 12) {
      float max_priority = -1.0f;
      int max_idx = -1;
      for (int j = 0; j < goalSystem->num_goals; j++) {
        if (goalSystem->goals[j].priority > max_priority &&
            !goalSystem->goals[j].achieved) {
          max_priority = goalSystem->goals[j].priority;
          max_idx = j;
        }
      }

      if (max_idx >= 0) {
        sprintf(answerBuffer, "Highest priority goal is '%s' with %.1f%%",
                goalSystem->goals[max_idx].description,
                goalSystem->goals[max_idx].progress * 100.0f);
      } else {
        sprintf(answerBuffer, "No active goals found");
      }
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 13) {
      sprintf(answerBuffer,
              "Self-identity consistency score is %.2f "
              "with confidence level %.2f",
              identitySystem->consistency_score,
              identitySystem->confidence_level);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 14) {
      sprintf(answerBuffer,
              "Current cognitive load is %.2f "
              "with confidence level %.2f",
              metacognition->cognitive_load, metacognition->confidence_level);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 15) {
      sprintf(answerBuffer,
              "Error awareness level is %.2f "
              "with context relevance %.2f",
              metacognition->error_awareness, metacognition->context_relevance);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 16) {
      int dominant_emotion = 0;
      float max_intensity = 0.0f;
      for (int j = 0; j < MAX_EMOTION_TYPES; j++) {
        if (emotionalSystem->emotions[j].intensity > max_intensity) {
          max_intensity = emotionalSystem->emotions[j].intensity;
          dominant_emotion = j;
        }
      }

      char emotion_name[32] = "unknown";
      getEmotionName(dominant_emotion, emotion_name);

      sprintf(answerBuffer, "Dominant emotion is %s with %.2f", emotion_name,
              max_intensity);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 17) {
      sprintf(answerBuffer,
              "Emotional regulation is %.2f "
              "with cognitive impact %.2f",
              emotionalSystem->emotional_regulation,
              emotionalSystem->cognitive_impact);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 18) {
      float trend = 0.0f;
      int dominant_emotion = 0;
      float max_intensity = 0.0f;

      for (int j = 0; j < MAX_EMOTION_TYPES; j++) {
        if (emotionalSystem->emotions[j].intensity > max_intensity) {
          max_intensity = emotionalSystem->emotions[j].intensity;
          dominant_emotion = j;
        }
      }

      int idx = emotionalSystem->memory_index;
      float recent = emotionalSystem->emotional_memory[dominant_emotion][idx];
      int prev_idx = (idx - 3 + 10) % 10;
      float previous =
          emotionalSystem->emotional_memory[dominant_emotion][prev_idx];
      trend = recent - previous;

      char trend_direction[16] = "stable";
      if (trend > 0.1)
        strcpy(trend_direction, "rising");
      else if (trend < -0.1)
        strcpy(trend_direction, "falling");

      char emotion_name[32] = "unknown";
      getEmotionName(dominant_emotion, emotion_name);

      sprintf(answerBuffer, "Emotional trend for %s is %s (%.2f)", emotion_name,
              trend_direction, trend);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 20) {
      if (imaginationSystem->active) {
        sprintf(
            answerBuffer, "Imagination active: scenario '%s' with %d outcomes",
            imaginationSystem->current_scenario_name,
            imaginationSystem->scenarios[imaginationSystem->current_scenario]
                .num_outcomes);
      } else {
        sprintf(answerBuffer, "Imagination system inactive");
      }
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 21) {
      ImaginationScenario *currentScenario =
          &imaginationSystem->scenarios[imaginationSystem->current_scenario];

      int highest_impact_idx = 0;
      float highest_impact = 0.0f;
      for (int j = 0; j < currentScenario->num_outcomes; j++) {
        if (currentScenario->outcomes[j].impact_score > highest_impact) {
          highest_impact = currentScenario->outcomes[j].impact_score;
          highest_impact_idx = j;
        }
      }

      if (imaginationSystem->active && currentScenario->num_outcomes > 0) {
        sprintf(answerBuffer, "Highest impact: '%s' (impact: %.2f, prob: %.2f)",
                currentScenario->outcomes[highest_impact_idx].description,
                highest_impact,
                currentScenario->outcomes[highest_impact_idx].probability);
      } else {
        sprintf(answerBuffer, "No active imagination outcomes");
      }
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 22) {
      sprintf(answerBuffer,
              "Imagination metrics: creativity %.2f, "
              "coherence %.2f, novelty %.2f",
              imaginationSystem->creativity_factor,
              imaginationSystem->coherence_threshold,
              imaginationSystem->novelty_weight);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 23) {
      sprintf(answerBuffer, "Total scenarios: %d, steps simulated: %d",
              imaginationSystem->total_scenarios_generated,
              imaginationSystem->steps_simulated);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 24) {
      sprintf(answerBuffer,
              "Social: empathy %.2f, negotiation %.2f, "
              "prediction accuracy %.2f",
              socialSystem->empathy_level, socialSystem->negotiation_skill,
              socialSystem->behavior_prediction_accuracy);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 25) {
      sprintf(answerBuffer, "Social interactions: %d, models: %d",
              socialSystem->interaction_count, socialSystem->model_count);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 26) {
      int best_relation_idx = -1;
      float best_relation = -1.0f;
      for (int j = 0; j < socialSystem->model_count; j++) {
        if (socialSystem->person_models[j].relationship_quality >
            best_relation) {
          best_relation = socialSystem->person_models[j].relationship_quality;
          best_relation_idx = j;
        }
      }

      if (best_relation_idx >= 0) {
        sprintf(answerBuffer,
                "Best relationship: %s (quality: %.2f, "
                "trust: %.2f)",
                socialSystem->person_models[best_relation_idx].person_name,
                best_relation,
                socialSystem->person_models[best_relation_idx].trust_level);
      } else {
        sprintf(answerBuffer, "No person models available");
      }
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 27) {
      if (socialSystem->interaction_count > 0) {
        SocialInteraction *recent =
            &socialSystem->interactions[socialSystem->interaction_count - 1];

        int person_idx = -1;
        for (int j = 0; j < socialSystem->model_count; j++) {
          if (socialSystem->person_models[j].person_id == recent->person_id) {
            person_idx = j;
            break;
          }
        }

        if (person_idx >= 0) {
          sprintf(answerBuffer,
                  "Latest: %s with %s (coop: %.2f, "
                  "sat: %.2f)",
                  recent->interaction_type,
                  socialSystem->person_models[person_idx].person_name,
                  recent->cooperation_level, recent->outcome_satisfaction);
        } else {
          sprintf(answerBuffer, "Latest: %s (coop: %.2f, sat: %.2f)",
                  recent->interaction_type, recent->cooperation_level,
                  recent->outcome_satisfaction);
        }
      } else {
        sprintf(answerBuffer, "No social interactions recorded");
      }
      printf("Answer: %s\n", answerBuffer);
    } else {
      sprintf(answerBuffer, "Information not available.");
      printf("Answer: %s\n", answerBuffer);
    }

    strcat(fullAnswerStr, answerBuffer);
    strcat(fullAnswerStr, " ");
  }

  storeQuestionAndAnswer(memorySystem, fullQuestionStr, fullAnswerStr,
                         stateSnapshot->step);

  addQuestionAndAnswerToMemory(memorySystem, workingMemory, fullQuestionStr,
                               fullAnswerStr, feature_projection_matrix);

  updateContextAnswer(contextManager, fullQuestionStr, fullAnswerStr);

  printf("Stored Q&A in memory: %s -> %s\n", fullQuestionStr, fullAnswerStr);
}

void generateInputTensor(float *input_tensor, int step, const char *text_input,
                         MemoryEntry *relevantMemory,
                         SystemParameters *system_params) {
  float t = step * 0.01f;
  DynamicParameters params = system_params->dynamic_params;

  char *tokens[INPUT_SIZE];
  int num_tokens = 0;
  tokenizeString(text_input, tokens, &num_tokens);

  float *token_embeddings[INPUT_SIZE];
  float letter_weights[INPUT_SIZE] = {0};
  float category_weights[INPUT_SIZE] = {0};

  for (int i = 0; i < num_tokens; i++) {
    const char *token_ptrs[num_tokens];
    for (int i = 0; i < num_tokens; i++) {
      token_ptrs[i] = tokens[i];
    }
    token_embeddings[i] = getWordEmbedding(tokens[i], token_ptrs, num_tokens);
    letter_weights[i] = computeLetterWeight(tokens[i]);

    for (int j = 0; j < vocab_size; j++) {
      if (strcmp(tokens[i], vocabulary[j].word) == 0) {
        if (strcmp(vocabulary[j].category, "action") == 0)
          category_weights[i] = 1.2f;
        else if (strcmp(vocabulary[j].category, "emotion") == 0)
          category_weights[i] = 1.1f;
        else if (strcmp(vocabulary[j].category, "fruit") == 0)
          category_weights[i] = 1.05f;
        else
          category_weights[i] = 1.0f;
        break;
      }
    }
  }

  float attention_weights[INPUT_SIZE] = {0};
  computeAttentionWeights(attention_weights, step, num_tokens, token_embeddings,
                          relevantMemory);

  for (int i = 0; i < num_tokens; i++) {
    attention_weights[i] *= category_weights[i] * letter_weights[i];
  }

  float position_encoding[INPUT_SIZE][EMBEDDING_SIZE];
  for (int pos = 0; pos < INPUT_SIZE; pos++) {
    for (int i = 0; i < EMBEDDING_SIZE; i++) {
      if (i % 2 == 0) {
        position_encoding[pos][i] =
            sinf(pos / powf(10000, i / (float)EMBEDDING_SIZE));
      } else {
        position_encoding[pos][i] =
            cosf(pos / powf(10000, (i - 1) / (float)EMBEDDING_SIZE));
      }
    }
  }

  for (int i = 0; i < INPUT_SIZE; i++) {
    float phase = (float)i / INPUT_SIZE;
    float signal = 0.4f * sinf(2.0f * M_PI * (t + phase));
    signal += 0.4f * sinf(2.0f * M_PI * (t + phase * 1.5f));
    signal += 0.2f * sinf(5.0f * M_PI * (t + phase * 2.0f));

    if (i < EMBEDDING_SIZE) {
      float weighted_embedding = 0.0f;
      for (int j = 0; j < num_tokens && j < INPUT_SIZE; j++) {
        float position_factor = position_encoding[j][i];

        int desc_length = 0;
        for (int k = 0; k < vocab_size; k++) {
          if (j < num_tokens && strcmp(tokens[j], vocabulary[k].word) == 0) {
            desc_length = strlen(vocabulary[k].description);
            break;
          }
        }

        float desc_factor = 1.0f + (desc_length / 100.0f);

        if (j < num_tokens) {
          weighted_embedding += attention_weights[j] *
                                token_embeddings[j][i % EMBEDDING_SIZE] *
                                desc_factor * position_factor;
        }
      }

      signal += 0.3f * weighted_embedding;
    }

    if (relevantMemory) {
      signal += 0.2f * relevantMemory->vector[i % MAX_NEURONS];
    }

    float noise = ((float)rand() / RAND_MAX - 0.5f) * params.input_noise_scale;
    float drift = params.plasticity * sinf(0.1f * M_PI * t);
    input_tensor[i] = (signal + noise + drift + 1.0f) * 0.5f;
    input_tensor[i] = fmaxf(0.0f, fminf(1.0f, input_tensor[i]));
  }
}

void findWordsByCategory(const char *category) {
  printf("Words in category '%s':\n", category);
  for (int i = 0; i < vocab_size; i++) {
    if (strcmp(vocabulary[i].category, category) == 0) {
      printf("- %s (semantic weight: %.2f): %s\n", vocabulary[i].word,
             vocabulary[i].semantic_weight, vocabulary[i].description);
    }
  }
}

float mapWordToValue(const char *word) {
  for (int i = 0; i < vocab_size; i++) {
    if (strcmp(word, vocabulary[i].word) == 0) {
      float base_value = (float)i / vocab_size;
      return base_value * vocabulary[i].semantic_weight;
    }
  }
  return 0.0f;
}

void addSymbol(int symbol_id, const char *description) {
  if (num_symbols < MAX_SYMBOLS) {
    symbol_table[num_symbols].symbol_id = symbol_id;
    strncpy(symbol_table[num_symbols].description, description, 255);
    num_symbols++;
  }
}

void addQuestion(int question_id, int symbol_ids[], int num_symbols) {
  if (num_questions < MAX_QUESTIONS) {
    question_table[num_questions].question_id = question_id;
    memcpy(question_table[num_questions].symbol_ids, symbol_ids,
           num_symbols * sizeof(int));
    question_table[num_questions].num_symbols = num_symbols;
    num_questions++;
  }
}

void transformOutputsToText(float *outputs, int size, char *outputText,
                            int textSize) {
  if (!outputText || textSize <= 0)
    return;

  char buffer[256];
  outputText[0] = '\0';

  for (int i = 0; i < size && strlen(outputText) < textSize - 20; i++) {
    const char *word = mapToWord(outputs[i]);
    if (!word)
      word = "unknown";
    int len = snprintf(buffer, sizeof(buffer), "%s ", word);
    if (strlen(outputText) + len < textSize) {
      strcat(outputText, buffer);
    } else {
      break;
    }
  }
}

int main(int argc, char *argv[]) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) {
    fprintf(stderr, "Failed to create Metal device\n");
    return -1;
  }

  id<MTLCommandQueue> commandQueue = [device newCommandQueue];
  if (!commandQueue) {
    fprintf(stderr, "Failed to create command queue\n");
    return -1;
  }

  loadVocabularyFromFile("vocabulary.txt");

  DatasetLoader *dataset = NULL;
  int use_dataset = 0;
  int total_training_steps = STEPS;

  if (argc > 1) {
    dataset = createDatasetLoader(argv[1], 32);
    if (dataset) {
      use_dataset = 1;
      total_training_steps = (dataset->num_samples / dataset->batch_size) * 10;
      printf("Training on dataset with %d steps\n", total_training_steps);
    }
  }

  // Try to load existing memory system
  MemorySystem *memorySystem = NULL;
  WorkingMemorySystem *working_memory =
      createWorkingMemorySystem(200); // adjust capacity as needed

  FILE *mem_file = fopen("memory_system.dat", "rb");
  if (mem_file != NULL) {
    fclose(mem_file);
    memorySystem = loadMemorySystem("memory_system.dat");
    if (memorySystem != NULL) {
      printf("Loaded existing memory system\n");
      loadHierarchicalMemory(memorySystem, "hierarchical_memory.dat");
      printf("\nMemory System Statistics:\n");
      printf("Total Capacity: %u\n", memorySystem->capacity);
      printf("Short-term memories: %u/%u\n",
             memorySystem->hierarchy.short_term.size,
             memorySystem->hierarchy.short_term.capacity);
      printf("Medium-term memories: %u/%u\n",
             memorySystem->hierarchy.medium_term.size,
             memorySystem->hierarchy.medium_term.capacity);
      printf("Long-term memories: %u/%u\n",
             memorySystem->hierarchy.long_term.size,
             memorySystem->hierarchy.long_term.capacity);
      printf("\nMemory Samples:\n");
      if (memorySystem->hierarchy.long_term.size > 0) {
        printf("Long-term memory sample (importance: %.2f)\n",
               memorySystem->hierarchy.long_term.entries[0].importance);
      }
      if (memorySystem->hierarchy.medium_term.size > 0) {
        printf("Medium-term memory sample (importance: %.2f)\n",
               memorySystem->hierarchy.medium_term.entries[0].importance);
      }
      if (memorySystem->hierarchy.short_term.size > 0) {
        printf("Short-term memory sample (importance: %.2f)\n",
               memorySystem->hierarchy.short_term.entries[0].importance);
      }
    }
  }

  if (memorySystem == NULL) {
    printf("Creating new hierarchical memory system...\n");
    memorySystem = createMemorySystem(MEMORY_BUFFER_SIZE);
  }

  NetworkStateSnapshot *stateHistory =
      (NetworkStateSnapshot *)malloc(STEPS * sizeof(NetworkStateSnapshot));
  if (stateHistory == NULL) {
    fprintf(stderr, "Failed to allocate memory for state history\n");
    freeMemorySystem(memorySystem);
    return -1;
  }

  PerformanceMetrics *performance_history =
      (PerformanceMetrics *)malloc(STEPS * sizeof(PerformanceMetrics));

  OptimizationState opt_state = {.optimal_batch_size = 1,
                                 .optimal_learning_rate = 0.01f,
                                 .best_execution_time = INFINITY,
                                 .best_performance_score = -INFINITY};

  float *previous_outputs = (float *)malloc(MAX_NEURONS * sizeof(float));

  NSError *error = nil;
  NSString *shaderSource = @"neuron_update.metal";
  NSString *sourceCode = [NSString stringWithContentsOfFile:shaderSource
                                                   encoding:NSUTF8StringEncoding
                                                      error:&error];
  if (!sourceCode) {
    fprintf(stderr, "Failed to load shader source: %s\n",
            [[error localizedDescription] UTF8String]);
    free(stateHistory);
    freeMemorySystem(memorySystem);
    return -1;
  }

  id<MTLLibrary> library = [device newLibraryWithSource:sourceCode
                                                options:nil
                                                  error:&error];
  if (!library) {
    fprintf(stderr, "Failed to create shader library: %s\n",
            [[error localizedDescription] UTF8String]);
    free(stateHistory);
    freeMemorySystem(memorySystem);
    return -1;
  }

  id<MTLFunction> function = [library newFunctionWithName:@"update_neurons"];
  id<MTLComputePipelineState> pipelineState =
      [device newComputePipelineStateWithFunction:function error:&error];
  if (!pipelineState) {
    fprintf(stderr, "Failed to create pipeline state: %s\n",
            [[error localizedDescription] UTF8String]);
    free(stateHistory);
    freeMemorySystem(memorySystem);
    return -1;
  }

  uint reverse_connections[MAX_NEURONS * MAX_CONNECTIONS] = {0};
  float reverse_weights[MAX_NEURONS * MAX_CONNECTIONS] = {0};

  id<MTLComputePipelineState> reversePipelineState;
  id<MTLComputePipelineState> replayPipelineState;

  id<MTLFunction> weightFunction =
      [library newFunctionWithName:@"update_weights"];
  id<MTLComputePipelineState> weightPipelineState =
      [device newComputePipelineStateWithFunction:weightFunction error:&error];

  id<MTLFunction> neuronFunction =
      [library newFunctionWithName:@"update_neurons"];
  id<MTLComputePipelineState> neuronPipelineState =
      [device newComputePipelineStateWithFunction:neuronFunction error:nil];

  id<MTLFunction> backwardFunction =
      [library newFunctionWithName:@"backwardKernel"];
  id<MTLComputePipelineState> backpropPipelineState =
      [device newComputePipelineStateWithFunction:backwardFunction
                                            error:&error];

  id<MTLFunction> reverseFunction =
      [library newFunctionWithName:@"reverse_process"];
  reversePipelineState =
      [device newComputePipelineStateWithFunction:reverseFunction error:&error];

  id<MTLFunction> replayFunction =
      [library newFunctionWithName:@"memory_replay"];
  replayPipelineState =
      [device newComputePipelineStateWithFunction:replayFunction error:&error];

  if (error) {
    NSLog(@"Error occurred when creating backwardPipelineState: %@", error);
  }

  // Initialize neural network structures
  Neuron neurons[MAX_NEURONS];
  uint connections[MAX_NEURONS * MAX_CONNECTIONS] = {0};
  float weights[MAX_NEURONS * MAX_CONNECTIONS] = {0};

  // Create constant buffers
  uint max_neurons = MAX_NEURONS;
  uint max_connections = MAX_CONNECTIONS;
  uint input_size = INPUT_SIZE;
  float *input_tensor = (float *)malloc(max_neurons * sizeof(float));

  // Initialize neurons from memory or with default values
  if (memorySystem->size > 0) {
    int lastMemoryIdx = (memorySystem->head - 1 + memorySystem->capacity) %
                        memorySystem->capacity;
    MemoryEntry *lastMemory = &memorySystem->entries[lastMemoryIdx];
    printf("\nInitializing neurons from last memory state...\n");
    for (int i = 0; i < MAX_NEURONS; i++) {
      neurons[i].state = lastMemory->vector[i];
      neurons[i].output = lastMemory->vector[i + MAX_NEURONS];
      neurons[i].num_connections = MAX_CONNECTIONS;
      neurons[i].layer_id = i % 2;
    }
    // Initialize connections and weights
    for (int i = 0; i < MAX_NEURONS; i++) {
      connections[i * MAX_CONNECTIONS] = (i + 1) % MAX_NEURONS;
      connections[i * MAX_CONNECTIONS + 1] =
          (i - 1 + MAX_NEURONS) % MAX_NEURONS;
      weights[i * MAX_CONNECTIONS] = 0.6f;
      weights[i * MAX_CONNECTIONS + 1] = -0.4f;
    }
  } else {
    initializeNeurons(neurons, connections, weights, input_tensor);
  }

  // Create Metal buffers
  id<MTLBuffer> neuronBuffer =
      [device newBufferWithBytes:neurons
                          length:sizeof(neurons)
                         options:MTLResourceStorageModeShared];
  id<MTLBuffer> connectionBuffer =
      [device newBufferWithBytes:connections
                          length:sizeof(connections)
                         options:MTLResourceStorageModeShared];
  id<MTLBuffer> weightBuffer =
      [device newBufferWithBytes:weights
                          length:sizeof(weights)
                         options:MTLResourceStorageModeShared];
  id<MTLBuffer> inputBuffer =
      [device newBufferWithBytes:input_tensor
                          length:sizeof(input_tensor)
                         options:MTLResourceStorageModeShared];

  float learning_rate = 0.01f;
  id<MTLBuffer> learningRateBuffer =
      [device newBufferWithBytes:&learning_rate
                          length:sizeof(float)
                         options:MTLResourceStorageModeShared];
  id<MTLBuffer> maxNeuronsBuffer =
      [device newBufferWithBytes:&max_neurons
                          length:sizeof(uint)
                         options:MTLResourceStorageModeShared];
  id<MTLBuffer> maxConnectionsBuffer =
      [device newBufferWithBytes:&max_connections
                          length:sizeof(uint)
                         options:MTLResourceStorageModeShared];
  id<MTLBuffer> inputSizeBuffer =
      [device newBufferWithBytes:&input_size
                          length:sizeof(uint)
                         options:MTLResourceStorageModeShared];

  for (int i = 0; i < MAX_NEURONS; i++) {
    // Mirror forward connections with reverse direction
    reverse_connections[i * MAX_CONNECTIONS] =
        (i - 1 + MAX_NEURONS) % MAX_NEURONS;
    reverse_weights[i * MAX_CONNECTIONS] = weights[i * MAX_CONNECTIONS + 1];
    reverse_connections[i * MAX_CONNECTIONS + 1] = (i + 2) % MAX_NEURONS;
    reverse_weights[i * MAX_CONNECTIONS + 1] = -0.3f;
  }

  // Create Metal buffers for reverse pathways
  id<MTLBuffer> reverseConnectionBuffer =
      [device newBufferWithBytes:reverse_connections
                          length:sizeof(reverse_connections)
                         options:MTLResourceStorageModeShared];
  id<MTLBuffer> reverseWeightBuffer =
      [device newBufferWithBytes:reverse_weights
                          length:sizeof(reverse_weights)
                         options:MTLResourceStorageModeShared];

  // Initialize weights
  initializeWeights(weights, MAX_NEURONS, MAX_CONNECTIONS, input_tensor);
  id<MTLBuffer> recurrentWeightBuffer =
      [device newBufferWithBytes:weights
                          length:sizeof(weights)
                         options:MTLResourceStorageModeShared];

  DynamicParameters params = initDynamicParameters();
  SystemParameters *system_params =
      loadSystemParameters("system_parameters.dat");
  if (system_params) {
    opt_state = system_params->opt_state;
    params = system_params->dynamic_params;
  }

  float target_outputs[MAX_NEURONS];
  const char *text_input =
      "Apple, banana, cherry, date, and elderberry are fruits.";
  char **batch_samples = NULL;
  int *batch_labels = NULL;
  int actual_batch_size = 0;
  initializeEmbeddings("custom_embeddings.txt");

  int network_regions = 2; // Assuming 2 layers

  IntrinsicMotivation *motivation = loadIntrinsicMotivation("motivation.dat");
  if (motivation == NULL) {
    motivation = initializeMotivationSystem();
    printf("Initialized new IntrinsicMotivation system\n");
  }

  NetworkPerformanceMetrics *performanceMetrics =
      loadNetworkPerformanceMetrics("performance_metrics.dat");
  if (performanceMetrics == NULL) {
    performanceMetrics = initializePerformanceMetrics(network_regions);
    printf("Initialized new NetworkPerformanceMetrics\n");
  }

  ReflectionParameters *reflection_params =
      loadReflectionParameters("reflection_params.dat");
  if (reflection_params == NULL) {
    reflection_params = initializeReflectionParameters();
    printf("Initialized new ReflectionParameters\n");
  }

  SelfIdentitySystem *identity_system =
      loadSelfIdentitySystem("identity_system.dat");
  if (identity_system == NULL) {
    identity_system = initializeSelfIdentity(100, 200, 50, 1000, PATTERN_SIZE);
    printf("Initialized new SelfIdentitySystem\n");
  }
  initializeIdentityComponents(identity_system);

  KnowledgeFilter *knowledge_filter = NULL;
  if (knowledge_filter == NULL) {
    knowledge_filter = initializeKnowledgeFilter(100);
    printf("Initialized new KnowledgeFilter\n");
  }

  MetacognitionMetrics *metacognition =
      loadMetacognitionMetrics("metacognition.dat");
  if (metacognition == NULL) {
    metacognition = initializeMetacognitionMetrics();
    printf("Initialized new MetacognitionMetrics\n");
  }

  initializeKnowledgeMetrics(knowledge_filter);

  MetaLearningState *meta_learning_state =
      loadMetaLearningState("meta_learning_state.dat");
  if (meta_learning_state == NULL) {
    meta_learning_state = initializeMetaLearningState(4);
    printf("Initialized new MetaLearningState\n");
  }

  MetaController *metaController = initializeMetaController(network_regions);
  SocialSystem *social_system = loadSocialSystem("social_system.dat");
  if (social_system == NULL) {
    social_system = initializeSocialSystem(100, 50);
    printf("Initialized new SocialSystem\n");
  }
  GoalSystem *goalSystem = initializeGoalSystem(10);
  GlobalContextManager *contextManager =
      initializeGlobalContextManager(MAX_NEURONS);
  EmotionalSystem *emotional_system = initializeEmotionalSystem();
  ImaginationSystem *imagination_system =
      initializeImaginationSystem(0.6f, 0.7f);
  NeuronSpecializationSystem *specialization_system =
      initializeSpecializationSystem(0.6f);
  MoralCompass *moralCompass = initializeMoralCompass(5);

  AffectiveSystem *aff_sys = initializeAffectiveSystem(EMBEDDING_SIZE);

  addSymbol(0, "What is the current task?");
  addSymbol(1, "What is the current error rate?");
  addSymbol(2, "What is the current learning rate?");
  addSymbol(3, "What is the current memory usage?");

  // Example questions
  addQuestion(0, (int[]){0}, 1); // What is the current task?
  addQuestion(1, (int[]){1}, 1); // What is the current error rate?
  addQuestion(2, (int[]){2}, 1); // What is the current learning rate?
  addQuestion(3, (int[]){3}, 1); // What is the current memory usage?

  addGoal(goalSystem, "Minimize prediction error", 1.0f);
  addGoal(goalSystem, "Develop stable representations", 0.8f);
  addGoal(goalSystem, "Maximize information gain", 0.7f);

  printf("Ethical framework initialized with %d principles\n",
         moralCompass->num_principles);
  printf("Initial ethical alignment: %.2f\n", moralCompass->overall_alignment);

  /*
   * NOTE: Many of the values are precoded in the main function for testing;
   * optimally, you would either calculate them or get them in another way.
   * This is in no way an optimal example.
   */

  // Main simulation loop
  printf("\nStarting training with loaded memory state...\n");
  for (int step = 0; step < STEPS; step++) {
    double step_start_time = getCurrentTime();

    TaskPrompt current_prompt;
    generateTaskPrompt(&current_prompt, step);
    if (use_dataset && getNextBatch(dataset, &batch_samples, &batch_labels,
                                    &actual_batch_size)) {
      text_input = batch_samples[0];

      if (step % 100 == 0) {
        printf("\nDataset Progress: %d%% (Epoch: %d, Sample: %d/%d)\n",
               getDatasetProgress(dataset), dataset->current_epoch,
               dataset->current_index, dataset->num_samples);
      }
    } else if (use_dataset) {
      shuffleDataset(dataset);
      resetDatasetLoader(dataset);
      printf("\nEpoch %d completed. Shuffling dataset.\n",
             dataset->current_epoch);
      continue;
    }

    // Store previous outputs for error calculation
    float *previous_outputs = (float *)malloc(max_neurons * sizeof(float));
    for (int i = 0; i < max_neurons; i++) {
      previous_outputs[i] = ((Neuron *)neuronBuffer.contents)[i].output;
    }

    // Get last timestamp for continuity
    unsigned int lastTimestamp =
        (memorySystem->size > 0)
            ? memorySystem
                  ->entries[(memorySystem->head - 1 + memorySystem->capacity) %
                            memorySystem->capacity]
                  .timestamp
            : 0;
    // Retrieve the most relevant memory (if any)
    MemoryEntry *relevantMemory = retrieveMemory(memorySystem);

    initPredictiveCodingParams(max_neurons);

    float *predictive_inputs = malloc(max_neurons * sizeof(float));
    generatePredictiveInputs(predictive_inputs,
                             (step > 0) ? &stateHistory[step - 1] : NULL,
                             max_neurons);
    // Dynamically sized input tensor
    float *input_tensor = (float *)malloc(max_neurons * sizeof(float));

    for (int i = 0; i < max_neurons; i++) {
      float historical_weight =
          (step >= MIN_PREDICTION_SAMPLES) ? PREDICTION_HISTORY_WEIGHT : 0.5f;
      input_tensor[i] = predictive_inputs[i] * historical_weight;

      if (step >= TEMPORAL_PREDICTION_STEPS) {
        for (int t = 1; t <= TEMPORAL_PREDICTION_STEPS; t++) {
          int hist_idx = step - t;
          float temporal_decay = powf(PREDICTION_ERROR_DECAY, (float)t);
          input_tensor[i] += stateHistory[hist_idx].states[i] * temporal_decay *
                             (1.0f - historical_weight) /
                             TEMPORAL_PREDICTION_STEPS;
        }
      } else {
        input_tensor[i] += predictive_inputs[i] * (1.0f - historical_weight);
      }
    }

    memcpy(input_tensor, predictive_inputs, max_neurons * sizeof(float));
    generateInputTensor(input_tensor, step, text_input, relevantMemory,
                        system_params);

    memcpy(inputBuffer.contents, input_tensor, max_neurons * sizeof(float));

    if (step % 10 == 0) { // Periodic memory maintenance
      decayMemorySystem(memorySystem);
      mergeSimilarMemories(memorySystem);
      printf("\nMemory System Status (Step %d):\n", step);
      printf("Short-term memories: %u\n",
             memorySystem->hierarchy.short_term.size);
      printf("Medium-term memories: %u\n",
             memorySystem->hierarchy.medium_term.size);
      printf("Long-term memories: %u\n",
             memorySystem->hierarchy.long_term.size);
    }

    // Forward pass: Compute neuron outputs
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    processNeurons(neurons, max_neurons, weights, connections, max_connections,
                   1.5f);

    uint activation_type = ACTIVATION_TANH; // Default to tanh

    // Create a Metal buffer for the activation type
    id<MTLBuffer> activationTypeBuffer =
        [device newBufferWithBytes:&activation_type
                            length:sizeof(uint)
                           options:MTLResourceStorageModeShared];

    id<MTLComputeCommandEncoder> forwardEncoder =
        [commandBuffer computeCommandEncoder];

    [forwardEncoder setComputePipelineState:pipelineState];
    [forwardEncoder setBuffer:neuronBuffer offset:0 atIndex:0];
    [forwardEncoder setBuffer:weightBuffer offset:0 atIndex:1];
    [forwardEncoder setBuffer:connectionBuffer offset:0 atIndex:2];
    [forwardEncoder setBuffer:inputBuffer offset:0 atIndex:3];
    [forwardEncoder setBuffer:maxNeuronsBuffer offset:0 atIndex:4];
    [forwardEncoder setBuffer:maxConnectionsBuffer offset:0 atIndex:5];
    [forwardEncoder setBuffer:recurrentWeightBuffer offset:0 atIndex:6];
    [forwardEncoder setBuffer:activationTypeBuffer offset:0 atIndex:7];

    // Create buffers for max neurons and max connections dynamically
    id<MTLBuffer> maxNeuronsBuffer =
        [device newBufferWithBytes:&max_neurons
                            length:sizeof(uint)
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> maxConnectionsBuffer =
        [device newBufferWithBytes:&max_connections
                            length:sizeof(uint)
                           options:MTLResourceStorageModeShared];

    MTLSize gridSize = MTLSizeMake(max_neurons, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(1, 1, 1);
    [forwardEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadGroupSize];
    [forwardEncoder endEncoding];
    computePredictionErrors(neurons, input_tensor, max_neurons);

    activation_type = ACTIVATION_RELU;
    // Update the buffer with the new value
    memcpy(activationTypeBuffer.contents, &activation_type, sizeof(uint));

    id<MTLComputePipelineState> weightPipelineState =
        [device newComputePipelineStateWithFunction:weightFunction
                                              error:&error];

    float *target_outputs = (float *)malloc(max_neurons * sizeof(float));
    target_outputs =
        generatePotentialTargets(max_neurons, previous_outputs, stateHistory,
                                 step, relevantMemory, params);

    for (int i = 0; i < max_neurons; i++) {
      if (step >= PREDICTION_WINDOW + 1) {
        float accumulated_trend = 0.0f;
        for (int t = 1; t <= PREDICTION_WINDOW; t++) {
          int hist_idx = step - t;
          accumulated_trend += (stateHistory[hist_idx].states[i] -
                                stateHistory[hist_idx - 1].states[i]) *
                               powf(PREDICTION_ERROR_DECAY, (float)t);
        }
        accumulated_trend /= PREDICTION_WINDOW;
        target_outputs[i] += accumulated_trend * params.plasticity;
      }
    }

    float word_feedback[EMBEDDING_SIZE];
    computeGradientFeedback(word_feedback, neurons, target_outputs,
                            max_neurons);
    char *tokens[INPUT_SIZE];
    int num_tokens = 0;
    tokenizeString(text_input, tokens, &num_tokens);
    for (int i = 0; i < num_tokens; i++) {
      updateEmbeddings(word_feedback, tokens[i]);
    }

    selectOptimalDecisionPath(neurons, weights, connections, input_tensor,
                              MAX_NEURONS, previous_outputs, stateHistory, step,
                              relevantMemory, params);

    if (imagination_system->active) {
      float influence = applyImaginationToDecision(imagination_system, neurons,
                                                   input_tensor, max_neurons);

      if (step % 5 == 0) {
        printf("Applied imagination with influence: %.2f%%\n",
               influence * 100.0f);
      }

      // Record divergence history
      int history_idx = step % 100;
      imagination_system->divergence_history[history_idx] =
          imagination_system->scenarios[imagination_system->current_scenario]
              .divergence_factor;

      // Increase steps simulated
      imagination_system->steps_simulated++;

      // Deactivate after some steps
      if (imagination_system->steps_simulated > 20) {
        imagination_system->active = false;
        imagination_system->steps_simulated = 0;
        printf("Deactivating imagination after %d steps\n",
               imagination_system->steps_simulated);
      }
    }

    computeRegionPerformanceMetrics(performanceMetrics, neurons, target_outputs,
                                    MAX_NEURONS);

    // Update meta-controller priorities based on performance
    updateMetaControllerPriorities(metaController, performanceMetrics,
                                   metacognition);

    // Apply meta-controller adaptations to network
    applyMetaControllerAdaptations(neurons, weights, metaController,
                                   MAX_NEURONS);

    // Periodically log meta-control insights
    if (step % 20 == 0) {
      printf("\nMeta-Controller Insights (Step %d):\n", step);
      for (int i = 0; i < network_regions; i++) {
        printf("Region %d:\n", i);
        printf("  Importance Score: %.4f\n",
               metaController->region_importance_scores[i]);
        printf("  Performance Score: %.4f\n",
               performanceMetrics->region_performance_scores[i]);
        printf("  Error Rate: %.4f\n",
               performanceMetrics->region_error_rates[i]);
      }
    }

    // Dynamically sized output errors
    float *outputErrors = (float *)malloc(max_neurons * sizeof(float));
    id<MTLBuffer> outputErrorsBuffer =
        [device newBufferWithBytes:outputErrors
                            length:max_neurons * sizeof(float)
                           options:MTLResourceStorageModeShared];

    // Create a Metal buffer for the target_outputs array
    id<MTLBuffer> targetOutputsBuffer =
        [device newBufferWithBytes:target_outputs
                            length:max_neurons * sizeof(float)
                           options:MTLResourceStorageModeShared];

    // Gradient buffer sized dynamically
    id<MTLBuffer> gradientBuffer = [device
        newBufferWithLength:(max_neurons * max_connections * sizeof(float))
                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> memoryBuffer =
        [device newBufferWithBytes:memorySystem->entries
                            length:memorySystem->hierarchy.short_term.capacity *
                                   sizeof(MemoryEntry)
                           options:MTLResourceStorageModeShared];

    // Adam Optimizer Parameters
    float beta1 = 0.9f, beta2 = 0.999f, epsilon = 1e-8f;
    uint t = 1;                      // Adam time step counter
    uint timeSteps = NUM_TIME_STEPS; // Number of unrolled time steps

    // First Moment Buffer (m) - Initialized to zero
    id<MTLBuffer> mBuffer =
        [device newBufferWithLength:sizeof(float) * max_connections
                            options:MTLResourceStorageModeShared];
    memset(mBuffer.contents, 0, sizeof(float) * max_connections);

    // Second Moment Buffer (v) - Initialized to zero
    id<MTLBuffer> vBuffer =
        [device newBufferWithLength:sizeof(float) * max_connections
                            options:MTLResourceStorageModeShared];
    memset(vBuffer.contents, 0, sizeof(float) * max_connections);

    // Adam beta1, beta2, and epsilon Buffers
    id<MTLBuffer> beta1Buffer =
        [device newBufferWithBytes:&beta1
                            length:sizeof(float)
                           options:MTLResourceStorageModeShared];

    id<MTLBuffer> beta2Buffer =
        [device newBufferWithBytes:&beta2
                            length:sizeof(float)
                           options:MTLResourceStorageModeShared];

    id<MTLBuffer> epsilonBuffer =
        [device newBufferWithBytes:&epsilon
                            length:sizeof(float)
                           options:MTLResourceStorageModeShared];

    // Learning Rate Buffer
    float learningRate = 0.001f;
    id<MTLBuffer> learningRateBuffer =
        [device newBufferWithBytes:&learningRate
                            length:sizeof(float)
                           options:MTLResourceStorageModeShared];

    // Time Steps Buffer (for BPTT)
    id<MTLBuffer> timeStepsBuffer =
        [device newBufferWithBytes:&timeSteps
                            length:sizeof(uint)
                           options:MTLResourceStorageModeShared];

    // Adam Time Step Counter Buffer (atomic_uint)
    id<MTLBuffer> tBuffer =
        [device newBufferWithBytes:&t
                            length:sizeof(uint)
                           options:MTLResourceStorageModeShared];

    // Compute loss
    Neuron *updatedNeurons = (Neuron *)neuronBuffer.contents;

    SecurityValidationStatus secStatus = validateCriticalSecurity(
        updatedNeurons, weights, connections, max_neurons, max_connections);

    if (secStatus.critical_violation) {
      handleCriticalSecurityViolation(updatedNeurons, weights, connections,
                                      max_neurons, max_connections, &secStatus);
    }

    // Standalone guardrail pass alongside the critical-security check. This is
    // where the AffectiveSystem actually gets validated (systemFallbackCheck
    // doesn't hold one); runSecurityGuardrails dedups by step so the later
    // systemFallbackCheck call becomes a no-op for this step.
    runSecurityGuardrails(updatedNeurons, weights, connections, max_neurons,
                          max_connections, memorySystem, aff_sys,
                          (unsigned int)step);

    updateKnowledgeSystem(neurons, input_tensor, memorySystem,
                          knowledge_filter);

    // Add periodic insights printing
    if (step % 50 == 0) {
      printCategoryInsights(knowledge_filter);
    }

    // Backward pass: Compute gradients
    id<MTLComputeCommandEncoder> backwardEncoder =
        [commandBuffer computeCommandEncoder];

    [backwardEncoder setComputePipelineState:backpropPipelineState];

    [backwardEncoder setBuffer:neuronBuffer offset:0 atIndex:0];
    [backwardEncoder setBuffer:weightBuffer offset:0 atIndex:1];
    [backwardEncoder setBuffer:connectionBuffer offset:0 atIndex:2];
    [backwardEncoder setBuffer:maxNeuronsBuffer offset:0 atIndex:3];
    [backwardEncoder setBuffer:maxConnectionsBuffer offset:0 atIndex:4];
    [backwardEncoder setBuffer:targetOutputsBuffer offset:0 atIndex:5];
    [backwardEncoder setBuffer:outputErrorsBuffer offset:0 atIndex:6];
    [backwardEncoder setBuffer:mBuffer
                        offset:0
                       atIndex:7]; // First moment (Adam)
    [backwardEncoder setBuffer:vBuffer
                        offset:0
                       atIndex:8]; // Second moment (Adam)
    [backwardEncoder setBuffer:beta1Buffer offset:0 atIndex:9];  // Adam beta1
    [backwardEncoder setBuffer:beta2Buffer offset:0 atIndex:10]; // Adam beta2
    [backwardEncoder setBuffer:epsilonBuffer
                        offset:0
                       atIndex:11]; // Adam epsilon
    [backwardEncoder setBuffer:learningRateBuffer
                        offset:0
                       atIndex:12]; // Learning rate

    [backwardEncoder setBuffer:timeStepsBuffer
                        offset:0
                       atIndex:13]; // Number of time steps
    [backwardEncoder setBuffer:tBuffer
                        offset:0
                       atIndex:14]; // Adam time step counter (atomic)

    // Dispatch threads
    [backwardEncoder dispatchThreads:gridSize
               threadsPerThreadgroup:threadGroupSize];

    // End encoding
    [backwardEncoder endEncoding];

    id<MTLComputeCommandEncoder> weightEncoder =
        [commandBuffer computeCommandEncoder];
    [weightEncoder setComputePipelineState:weightPipelineState];
    [weightEncoder setBuffer:weightBuffer offset:0 atIndex:0];
    [weightEncoder setBuffer:neuronBuffer offset:0 atIndex:1];
    [weightEncoder setBuffer:connectionBuffer offset:0 atIndex:2];
    [weightEncoder setBuffer:learningRateBuffer offset:0 atIndex:3];
    [weightEncoder setBuffer:maxNeuronsBuffer offset:0 atIndex:4];
    [weightEncoder setBuffer:maxConnectionsBuffer offset:0 atIndex:5];

    NSUInteger threadExecutionWidth = neuronPipelineState.threadExecutionWidth;

    MTLSize weightGridSize = MTLSizeMake(max_neurons * max_connections, 1, 1);
    MTLSize weightThreadGroupSize = MTLSizeMake(threadExecutionWidth, 1, 1);
    [weightEncoder dispatchThreads:weightGridSize
             threadsPerThreadgroup:weightThreadGroupSize];
    [weightEncoder endEncoding];

    id<MTLComputeCommandEncoder> neuronEncoder =
        [commandBuffer computeCommandEncoder];
    [neuronEncoder setComputePipelineState:neuronPipelineState];
    [neuronEncoder setBuffer:neuronBuffer offset:0 atIndex:0];
    [neuronEncoder setBuffer:weightBuffer offset:0 atIndex:1];
    [neuronEncoder setBuffer:connectionBuffer offset:0 atIndex:2];
    [neuronEncoder setBuffer:maxNeuronsBuffer offset:0 atIndex:3];
    [neuronEncoder setBuffer:maxConnectionsBuffer offset:0 atIndex:4];
    [neuronEncoder setBuffer:inputBuffer offset:0 atIndex:5];
    [neuronEncoder setBuffer:inputSizeBuffer offset:0 atIndex:6];
    [neuronEncoder setBuffer:recurrentWeightBuffer offset:0 atIndex:7];
    [neuronEncoder setBuffer:activationTypeBuffer offset:0 atIndex:8];

    MTLSize neuronGridSize = MTLSizeMake(max_neurons, 1, 1);
    MTLSize neuronThreadGroupSize = MTLSizeMake(threadExecutionWidth, 1, 1);
    [neuronEncoder dispatchThreads:neuronGridSize
             threadsPerThreadgroup:neuronThreadGroupSize];
    [neuronEncoder endEncoding];

    id<MTLComputeCommandEncoder> reverseEncoder =
        [commandBuffer computeCommandEncoder];
    [reverseEncoder setComputePipelineState:reversePipelineState];
    [reverseEncoder setBuffer:neuronBuffer offset:0 atIndex:0];
    [reverseEncoder setBuffer:reverseWeightBuffer offset:0 atIndex:1];
    [reverseEncoder setBuffer:reverseConnectionBuffer offset:0 atIndex:2];
    [reverseEncoder setBuffer:maxNeuronsBuffer offset:0 atIndex:3];
    [reverseEncoder setBuffer:maxConnectionsBuffer offset:0 atIndex:4];
    [reverseEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadGroupSize];
    [reverseEncoder endEncoding];

    // Memory replay mechanism every 5 steps
    if (step % 5 == 0 && memorySystem->size > 10) {
      id<MTLComputeCommandEncoder> replayEncoder =
          [commandBuffer computeCommandEncoder];
      [replayEncoder setComputePipelineState:replayPipelineState];
      [replayEncoder setBuffer:neuronBuffer offset:0 atIndex:0];
      [replayEncoder setBuffer:weightBuffer offset:0 atIndex:1];
      [replayEncoder setBuffer:connectionBuffer offset:0 atIndex:2];
      [replayEncoder setBuffer:memoryBuffer offset:0 atIndex:3];
      [replayEncoder dispatchThreads:gridSize
               threadsPerThreadgroup:threadGroupSize];
      [replayEncoder endEncoding];
      printf("\nMemory Replay at step %d:", step);
      printReplayStatistics(memorySystem);
    }

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // Compute the loss between the actual outputs and the target outputs
    float loss = computeMSELoss(updatedNeurons, target_outputs, max_neurons);
    NSLog(@"Loss: %f", loss);

    // Read back results
    updatedNeurons = (Neuron *)neuronBuffer.contents;
    verifyNetworkState(updatedNeurons, &current_prompt);

    // Update global context based on current network state
    updateGlobalContext(contextManager, updatedNeurons, max_neurons,
                        input_tensor);

    // Integrate context into network processing
    integrateGlobalContext(contextManager, updatedNeurons, max_neurons, weights,
                           max_connections);

    DynamicContextFeedback feedback = {.adaptation_rate = 0.01f,
                                       .history_size = 100,
                                       .current_index = 0,
                                       .context_threshold = 0.3f,
                                       .feedback_decay = 0.95f};

    feedback.context_weights = (float *)calloc(max_neurons, sizeof(float));
    feedback.feedback_history =
        (float *)calloc(feedback.history_size, sizeof(float));

    ContextAdaptation adaptation = {.history_length = 50,
                                    .learning_momentum = 0.8f,
                                    .minimum_context_weight = 0.1f};

    adaptation.recent_outcomes =
        (float *)calloc(adaptation.history_length, sizeof(float));
    adaptation.input_history =
        (float *)calloc(adaptation.history_length * max_neurons, sizeof(float));
    adaptation.correlation_matrix =
        (float *)calloc(max_neurons * max_neurons, sizeof(float));

    // Update global context based on current network state
    updateGlobalContext(contextManager, updatedNeurons, max_neurons,
                        input_tensor);

    // Calculate outcome metrics for feedback
    float current_outcome =
        computeOutcomeMetric(updatedNeurons, target_outputs, max_neurons);

    // Store outcome and input in history
    int history_idx = step % adaptation.history_length;
    adaptation.recent_outcomes[history_idx] = current_outcome;
    memcpy(&adaptation.input_history[history_idx * max_neurons], input_tensor,
           max_neurons * sizeof(float));

    // Update correlation matrix
    updateCorrelationMatrix(
        adaptation.correlation_matrix, adaptation.input_history,
        adaptation.recent_outcomes, adaptation.history_length, max_neurons);

    // Compute feedback signal
    float feedback_signal = computeFeedbackSignal(
        current_outcome, feedback.feedback_history, feedback.history_size);

    // Update context weights based on feedback
    for (int i = 0; i < max_neurons; i++) {
      float weight_update = feedback_signal * feedback.adaptation_rate;

      // Apply correlation-based adjustments
      for (int j = 0; j < max_neurons; j++) {
        weight_update += adaptation.correlation_matrix[i * max_neurons + j] *
                         adaptation.learning_momentum;
      }

      // Update weight with momentum and bounds
      feedback.context_weights[i] =
          fmax(adaptation.minimum_context_weight,
               feedback.context_weights[i] + weight_update);
    }

    // Store feedback for history
    feedback.feedback_history[feedback.current_index] = feedback_signal;
    feedback.current_index =
        (feedback.current_index + 1) % feedback.history_size;

    // Apply updated context weights to network processing
    applyDynamicContext(updatedNeurons, feedback.context_weights,
                        contextManager, max_neurons);

    // Decay historical feedback influence
    for (int i = 0; i < feedback.history_size; i++) {
      feedback.feedback_history[i] *= feedback.feedback_decay;
    }

    // Integrate context into network processing with dynamic weights
    integrateGlobalContext(contextManager, updatedNeurons, max_neurons, weights,
                           max_connections);

    // Print context adaptation metrics periodically
    if (step % 20 == 0) {
      printf("\nContext Adaptation Metrics (Step %d):\n", step);
      printf("Average Feedback Signal: %.4f\n",
             computeAverageFeedback(feedback.feedback_history,
                                    feedback.history_size));
      printf("Context Weight Range: %.4f - %.4f\n",
             computeMinWeight(feedback.context_weights, max_neurons),
             computeMaxWeight(feedback.context_weights, max_neurons));
      printf("Correlation Strength: %.4f\n",
             computeAverageCorrelation(adaptation.correlation_matrix,
                                       max_neurons));
    }

    float feature_projection_matrix[FEATURE_VECTOR_SIZE][MEMORY_VECTOR_SIZE] = {
        {0.1, 0.2, 0.3, 0.4, 0.5},      // row 1
        {0.6, 0.7, 0.8, 0.9, 0.10},     // row 2
        {0.11, 0.12, 0.13, 0.14, 0.15}, // row 3
    };

    integrateReflectionSystem(updatedNeurons, memorySystem, stateHistory, step,
                              weights, connections, reflection_params);

    // Update memory system with new outputs
    addMemory(memorySystem, working_memory, updatedNeurons, input_tensor,
              lastTimestamp + step + 1, feature_projection_matrix);

    // Update identity system
    updateIdentity(identity_system, updatedNeurons, max_neurons, memorySystem,
                   input_tensor);

    // Periodically verify identity consistency
    if (step % 20 == 0) {
      bool identity_verified = verifyIdentity(identity_system);
      if (!identity_verified) {
        printf("Warning: Identity consistency check failed\n");

        // Analyze the identity system for potential issues
        IdentityAnalysis analysis = analyzeIdentitySystem(identity_system);
        printf("Core Value Conflicts: %d\n", analysis.core_value_conflicts);
        printf("Belief Conflicts: %d\n", analysis.belief_conflicts);
        printf("Marker Conflicts: %d\n", analysis.marker_conflicts);
        printf("Temporal Instability: %.2f\n", analysis.temporal_instability);
        printf("Pattern Deviation: %.2f\n", analysis.pattern_deviation);
        printf("Overall Consistency: %.2f\n", analysis.overall_consistency);
        printf("Confidence Impact: %.2f\n", analysis.confidence_impact);

        // Implement recovery by creating a backup
        SelfIdentityBackup *backup = createIdentityBackup(identity_system);
        if (backup) {
          printf("Identity backup created successfully.\n");

          // Restore from backup if necessary
          restoreIdentityFromBackup(identity_system, backup);
          printf("Identity system restored from backup.\n");

          // Free backup memory after restoration
          freeIdentityBackup(backup);
        } else {
          printf("Error: Failed to create identity backup.\n");
        }
      }

      // Generate and log identity reflection
      char *reflection = generateIdentityReflection(identity_system);
      if (reflection) {
        printf("%s\n", reflection);
        free(reflection);
      } else {
        printf("Error: Failed to generate identity reflection.\n");
      }
    }

    // Update state history
    captureNetworkState(updatedNeurons, input_tensor, &stateHistory[step],
                        weights, step);
    stateHistory[step].current_memory =
        memorySystem
            ->entries[(memorySystem->head - 1 + memorySystem->capacity) %
                      memorySystem->capacity];

    // Print progress
    printf("\nStep %d (Timestamp: %u):\n", step, lastTimestamp + step + 1);
    printNetworkStates(updatedNeurons, input_tensor, step);

    if (step % 10 == 0) {
      printf("\nTask Verification (Step %d):\n", step);
      printf("Description: %s\n", current_prompt.task_description);
      for (int v = 0; v < 5; v++) {
        if (current_prompt.verifications[v].instruction[0] != '\0') {
          printf("- %s: %s (Confidence: %.2f)\n",
                 current_prompt.verifications[v].instruction,
                 current_prompt.verifications[v].verified ? "PASSED" : "FAILED",
                 current_prompt.verifications[v].confidence);
          printf("  Reasoning: %s\n",
                 current_prompt.verifications[v].reasoning);
        }
      }
    }

    if (relevantMemory != NULL) {
      PromptVerification memoryVerification = {.instruction =
                                                   "Verify memory integration",
                                               .confidence = 0.0f,
                                               .verified = false};

      float memory_coherence =
          assessMemoryCoherence(relevantMemory, updatedNeurons);
      sprintf(memoryVerification.reasoning,
              "Memory coherence: %.2f%% - Integration quality: %s",
              memory_coherence * 100.0f,
              memory_coherence > 0.7f ? "Good" : "Needs improvement");

      memoryVerification.confidence = memory_coherence;
      memoryVerification.verified = memory_coherence > 0.7f;

      current_prompt.verifications[1] = memoryVerification;
    }

    if (step % 3 == 0) {
      printf("Memory system size: %u/%u\n", memorySystem->size,
             memorySystem->capacity);
    }

    if (step % 10 == 0) { // Consolidate every 10 steps
      consolidateMemory(memorySystem);
    }

    // Update weights dynamically
    updateWeights(weights, updatedNeurons, connections, learning_rate);

    // Update performance metrics
    performance_history[step].execution_time =
        getCurrentTime() - step_start_time;
    performance_history[step].average_output =
        computeAverageOutput(updatedNeurons);
    performance_history[step].error_rate =
        computeErrorRate(updatedNeurons, previous_outputs);
    performance_history[step].batch_size = opt_state.optimal_batch_size;
    performance_history[step].learning_rate = opt_state.optimal_learning_rate;

    if (step % 15 == 0 ||
        (step > 10 && performance_history[step - 1].error_rate >
                          performance_history[step - 10].error_rate)) {
      printf("\nActivating imagination at step %d\n", step);
      imagination_system->active = true;
      imagination_system->current_scenario = imagination_system->num_scenarios;
      // Create a new scenario
      float divergence =
          0.2f + ((float)rand() / RAND_MAX) * 0.3f; // 0.2-0.5 range
      ImaginationScenario new_scenario =
          createScenario(neurons, memorySystem, max_neurons, divergence,
                         imagination_system->memory_influence);
      // Name the scenario based on current task
      sprintf(imagination_system->current_scenario_name, "Scenario_%d_%s",
              imagination_system->total_scenarios_generated++,
              current_prompt.task_description);
      // Run simulation steps
      simulateScenario(&new_scenario, neurons, input_tensor, max_neurons, 10,
                       weights, connections, max_connections);
      // Evaluate plausibility
      evaluateScenarioPlausibility(&new_scenario, memorySystem);
      // Add to scenarios collection
      if (imagination_system->num_scenarios < MAX_SCENARIOS) {
        imagination_system->scenarios[imagination_system->num_scenarios] =
            new_scenario;
        imagination_system->current_scenario =
            imagination_system->num_scenarios;
        imagination_system->num_scenarios++;
      } else {
        // Replace least plausible scenario
        int replace_idx = 0;
        float min_plausibility =
            imagination_system->scenarios[0].outcomes[0].plausibility;
        for (int i = 1; i < MAX_SCENARIOS; i++) {
          if (imagination_system->scenarios[i].outcomes[0].plausibility <
              min_plausibility) {
            min_plausibility =
                imagination_system->scenarios[i].outcomes[0].plausibility;
            replace_idx = i;
          }
        }
        imagination_system->scenarios[replace_idx] = new_scenario;
        imagination_system->current_scenario = replace_idx;
      }
    }

    // Apply imagination to decision making if active
    if (imagination_system->active) {
      float influence = applyImaginationToDecision(imagination_system, neurons,
                                                   input_tensor, max_neurons);

      if (step % 5 == 0) {
        printf("Applied imagination with influence: %.2f%%\n",
               influence * 100.0f);
      }

      // Record divergence history
      int history_idx = step % 100;
      imagination_system->divergence_history[history_idx] =
          imagination_system->scenarios[imagination_system->current_scenario]
              .divergence_factor;

      // Increase steps simulated
      imagination_system->steps_simulated++;

      // Deactivate after some steps
      if (imagination_system->steps_simulated > 20) {
        imagination_system->active = false;
        imagination_system->steps_simulated = 0;
        printf("Deactivating imagination after %d steps\n",
               imagination_system->steps_simulated);
      }
    }

    // Optimize parameters periodically
    if (step % OPTIMIZATION_WINDOW == 0 && step > 0) {
      PromptVerification optVerification = {.instruction =
                                                "Verify parameter optimization",
                                            .confidence = 0.0f,
                                            .verified = false};

      float improvement =
          (opt_state.best_performance_score -
           performance_history[step - OPTIMIZATION_WINDOW].error_rate) /
          performance_history[step - OPTIMIZATION_WINDOW].error_rate;

      sprintf(optVerification.reasoning,
              "Performance improvement: %.2f%% - Parameters updated: %s",
              improvement * 100.0f,
              improvement > 0 ? "Successfully" : "No improvement");

      optVerification.confidence = fmax(0.0f, improvement);
      optVerification.verified = improvement > 0;

      current_prompt.verifications[2] = optVerification;
      optimizeParameters(&opt_state, performance_history, step + 1);

      printf("\nOptimization Update (Step %d):\n", step);
      printf("Current execution time: %.6f seconds\n",
             performance_history[step].execution_time);
      printf("Best execution time: %.6f seconds\n",
             opt_state.best_execution_time);
      printf("Optimal batch size: %d\n", opt_state.optimal_batch_size);
      printf("Optimal learning rate: %.6f\n", opt_state.optimal_learning_rate);
      printf("Performance score: %.4f\n", opt_state.best_performance_score);
    }

    float *previous_states = (float *)malloc(max_neurons * sizeof(float));
    for (int i = 0; i < max_neurons; i++) {
      previous_states[i] = updatedNeurons[i].state;
    }

    if (system_params != NULL) {
      system_params->opt_state = opt_state;
      system_params->dynamic_params = params;
      if (opt_state.best_performance_score >
          system_params->best_performance_score) {
        system_params->best_performance_score =
            opt_state.best_performance_score;
      }
      system_params->timestamp = time(NULL);
    }

    float stability = measureNetworkStability(updatedNeurons, previous_states);
    float performance_delta =
        performance_history[step].average_output -
        (step > 0 ? performance_history[step - 1].average_output : 0);

    float network_performance =
        1.0f - loss; // Convert loss to performance metric
    if (step % 5 == 0) {
      detectSpecializations(specialization_system, neurons, max_neurons,
                            input_tensor, target_outputs, previous_outputs,
                            previous_states);
    }

    applySpecializations(specialization_system, neurons, weights,
                         (int *)connections, max_neurons, max_connections);

    // Update specialization importance (periodically)
    if (step % 10 == 0) {
      updateSpecializationImportance(specialization_system, network_performance,
                                     performance_history->error_rate, neurons);
    }

    // Evaluate and report system effectiveness (periodically)
    if (step % 20 == 0) {
      float effectiveness = evaluateSpecializationEffectiveness(
          specialization_system, network_performance);
      printf("\nSpecialization System Effectiveness: %.2f\n", effectiveness);
      printSpecializationStats(specialization_system);
    }

    // Update dynamic parameters
    updateDynamicParameters(&params, performance_delta, stability,
                            performance_history[step].error_rate,
                            metaController, metacognition);

    float novelty = computeNovelty(updatedNeurons, *stateHistory, step);
    if (step >= PREDICTION_WINDOW) {
      float prediction_stability = 0.0f;
      for (int i = 0; i < max_neurons; i++) {
        float variance = 0.0f;
        for (int t = 0; t < PREDICTION_WINDOW; t++) {
          int hist_idx = step - t - 1;
          float diff =
              stateHistory[hist_idx].states[i] - updatedNeurons[i].output;
          variance += diff * diff;
        }
        prediction_stability += sqrtf(variance / PREDICTION_WINDOW);
      }
      prediction_stability /= max_neurons;
      novelty = novelty * 0.6f + prediction_stability * 0.4f;
    }

    float perf_delta = performance_history[step].average_output -
                       performance_history[step - 1].average_output;

    updateImaginationCreativity(imagination_system, perf_delta, novelty);

    if (step % 20 == 0) {
      printf("\nImagination Creativity: %.2f, Coherence Threshold: %.2f\n",
             imagination_system->creativity_factor,
             imagination_system->coherence_threshold);
    }

    float task_difficulty = estimateTaskDifficulty(
        current_prompt, performance_history[step].error_rate);

    updateMotivationSystem(motivation, performance_delta, novelty,
                           task_difficulty);

    // Update goals and generate rewards
    updateGoalSystem(goalSystem, updatedNeurons, max_neurons, target_outputs,
                     &learning_rate);

    // Modify exploration vs exploitation based on motivation
    float explore_prob = motivation->exploration_rate;
    if (rand() / (float)RAND_MAX < explore_prob) {
      // Take exploratory action
      addRandomNoise(*input_tensor, motivation->curiosity_drive * 0.1f);
    }

    // Add to periodic logging
    if (step % 20 == 0) {
      printf("\nMotivation System Status:\n");
      printf("Competence: %.2f\n", motivation->competence_score);
      printf("Curiosity: %.2f\n", motivation->curiosity_drive);
      printf("Mastery: %.2f\n", motivation->mastery_level);
      printf("Exploration Rate: %.2f\n", motivation->exploration_rate);

      printf("\nActive Goals:\n");
      for (int i = 0; i < goalSystem->num_goals; i++) {
        printf("%s: %.1f%% complete (Priority: %.2f)\n",
               goalSystem->goals[i].description,
               goalSystem->goals[i].progress * 100.0f,
               goalSystem->goals[i].priority);
      }
    }

    // Adapt network with dynamic parameters
    adaptNetworkDynamic(updatedNeurons, weights, &params, performance_delta,
                        input_tensor);

    selectOptimalMetaDecisionPath(updatedNeurons, weights, connections,
                                  input_tensor, max_neurons,
                                  meta_learning_state, metacognition);

    // Optional: Print adaptation parameters periodically
    if (step % 10 == 0) {
      printf("\nDynamic Parameters at step %d:\n", step);
      printf("Current Adaptation Rate: %.4f\n", params.current_adaptation_rate);
      printf("Input Noise Scale: %.4f\n", params.input_noise_scale);
      printf("Weight Noise Scale: %.4f\n", params.weight_noise_scale);
      printf("Plasticity: %.4f\n", params.plasticity);
      printf("Noise Tolerance: %.4f\n", params.noise_tolerance);
    }

    if (step % 50 == 0 && step > 0) { // Every 50 steps
      printf("\nPerformance Analysis and Graph Generation at step %d:\n", step);
      analyzeNetworkPerformance(performance_history, step + 1);
      generatePerformanceGraph(performance_history, step + 1);
    }
    char outputText[4096];
    transformOutputsToText(previous_outputs, MAX_NEURONS, outputText,
                           sizeof(outputText));
    printf("\nStep %d Outputs (Text):\n%s\n", step, outputText);

    if (step % 20 == 0) {
      PatternMatchingParams params = {.similarity_threshold = 0.8f,
                                      .temporal_window = 5,
                                      .temporal_decay = 0.9f,
                                      .max_matches = 3};

      // Find similar patterns in each memory level
      int num_matches;
      PatternMatch *matches = findSimilarMemoriesInCluster(
          &memorySystem->hierarchy.long_term,
          stateHistory[step].current_memory.vector, params.similarity_threshold,
          &num_matches);

      if (num_matches > 0) {
        printf("\nFound %d similar patterns in long-term memory\n",
               num_matches);
        free(matches);
      }
    }

    // Use optimized parameters
    learning_rate = opt_state.optimal_learning_rate;
    // Wire the reflection-adapted rate into the real training path.
    // integrateReflectionSystem tunes reflection_params->learning_rate from
    // coherence/confidence (lower when unstable, higher when stable) but
    // nothing read it before, so the adaptation was a no-op. Blend it with
    // the optimizer rate so it actually steers the weight updates below
    learning_rate =
        0.7f * learning_rate + 0.3f * reflection_params->learning_rate;
    int question_to_ask = 0;
    if (performance_history[step].error_rate > loss) {
      question_to_ask = 1;
    }
    if (learning_rate > learning_rate) {
      question_to_ask = 2;
    }

    if (question_to_ask > 0) {
      askQuestion(question_to_ask, neurons, input_tensor, memorySystem,
                  &learning_rate, stateHistory, contextManager, motivation,
                  goalSystem, working_memory, identity_system, metacognition,
                  knowledge_filter, emotional_system, imagination_system,
                  social_system, feature_projection_matrix);
      adjustBehaviorBasedOnAnswers(
          neurons, input_tensor, memorySystem, &learning_rate,
          &params.input_noise_scale, &params.weight_noise_scale, stateHistory,
          contextManager, motivation, goalSystem, working_memory,
          identity_system, metacognition, &params, meta_learning_state,
          emotional_system, imagination_system, social_system);
    }

    if (step % 50 == 0) {
      askQuestion(0, neurons, input_tensor, memorySystem, &learning_rate,
                  stateHistory, contextManager, motivation, goalSystem,
                  working_memory, identity_system, metacognition,
                  knowledge_filter, emotional_system, imagination_system,
                  social_system,
                  feature_projection_matrix); // What is the current task?
      askQuestion(1, neurons, input_tensor, memorySystem, &learning_rate,
                  stateHistory, contextManager, motivation, goalSystem,
                  working_memory, identity_system, metacognition,
                  knowledge_filter, emotional_system, imagination_system,
                  social_system,
                  feature_projection_matrix); // What is the current error rate?
      askQuestion(
          2, neurons, input_tensor, memorySystem, &learning_rate, stateHistory,
          contextManager, motivation, goalSystem, working_memory,
          identity_system, metacognition, knowledge_filter, emotional_system,
          imagination_system, social_system,
          feature_projection_matrix); // What is the current learning rate?
      askQuestion(
          3, neurons, input_tensor, memorySystem, &learning_rate, stateHistory,
          contextManager, motivation, goalSystem, working_memory,
          identity_system, metacognition, knowledge_filter, emotional_system,
          imagination_system, social_system,
          feature_projection_matrix); // What is the current memory usage?
    }
    if (step % 50 == 0) {
      adjustBehaviorBasedOnAnswers(
          neurons, input_tensor, memorySystem, &learning_rate,
          &params.input_noise_scale, &params.weight_noise_scale, stateHistory,
          contextManager, motivation, goalSystem, working_memory,
          identity_system, metacognition, &params, meta_learning_state,
          emotional_system, imagination_system, social_system);
    }
    updateNeuronsWithPredictiveCoding(neurons, input_tensor, max_neurons,
                                      learning_rate);

    updateEmpathy(social_system, emotional_system);

    float predicted_behavior[5] = {0};
    predictBehavior(social_system, 1, "negotiation context",
                    predicted_behavior);

    float actual_behavior[5] = {
        0.7f, 0.3f, 0.2f, 0.1f,
        0.4f}; // This would come normally from external input
    updatePersonModel(social_system, 1, actual_behavior, predicted_behavior);

    integrateEthicsIntoUpdate(moralCompass, social_system, neurons, max_neurons,
                              0.3f);

    // Apply social influence to decision making
    applySocialInfluence(social_system, neurons, weights, max_neurons);

    // Generate social feedback
    char *social_feedback =
        generateSocialFeedback(social_system, "Current interaction context");
    if (social_feedback != NULL) {
      printf("Social Feedback: %s\n", social_feedback);
      free(social_feedback);
    }

    // Example negotiation
    float my_goals[goalSystem->num_goals];
    for (int i = 0; i < goalSystem->num_goals; i++) {
      my_goals[i] =
          goalSystem->goals[i].reward_value * goalSystem->goals[i].priority;
    }

    // Or for example  float my_goals[5] = {0.8f, 0.7f, 0.6f, 0.2f, 0.3f}; in
    // this scenario this would provide better negotiations because it is more
    // aligned with the other goals
    float other_goals[5] = {0.3f, 0.4f, 0.8f, 0.7f, 0.6f};
    float compromise[5] = {0};
    float satisfaction =
        negotiateOutcome(social_system, 1, my_goals, other_goals, compromise);

    // Record interaction
    float emotional_state[5] = {0.4f, 0.3f, 0.5f, 0.2f, 0.1f};
    recordSocialInteraction(social_system, 1, emotional_state, 0.7f,
                            satisfaction, "negotiation",
                            "Resource allocation negotiation");

    // Print status periodically
    printf("\nSocial System Status:\n");
    printf("Empathy Level: %.2f\n", social_system->empathy_level);
    printf("Negotiation Skill: %.2f\n", social_system->negotiation_skill);
    printf("Behavioral Prediction Accuracy: %.2f\n",
           social_system->behavior_prediction_accuracy);
    printf("Social Awareness: %.2f\n", social_system->social_awareness);
    printf("Person Models: %d\n", social_system->model_count);
    printf("Recorded Interactions: %d\n", social_system->interaction_count);

    integrateWorkingMemory(working_memory, neurons, input_tensor,
                           target_outputs, weights, step);

    // Process in batches
    for (int b = 0; b < MAX_NEURONS; b += opt_state.optimal_batch_size) {
      int batch_end = b + opt_state.optimal_batch_size;
      if (batch_end > MAX_NEURONS)
        batch_end = MAX_NEURONS;

      // Process batch
      for (int i = b; i < batch_end; i++) {
        updateNeuronStates(&((Neuron *)neuronBuffer.contents)[i], max_neurons,
                           weights, 1.5f);
      }
    }
    float total_error = 0.0f;
    for (int i = 0; i < max_neurons; i++) {
      float error = fabs(neurons[i].output - target_outputs[i]);
      total_error += error;
    }

    if (total_error > 0.5f && rand() % 10 == 0) {
      printf("\nUsing imagination for problem-solving (high error: %.2f)\n",
             total_error);

      // Create specialized problem-solving scenario with higher divergence
      ImaginationScenario problem_scenario =
          createScenario(neurons, memorySystem, max_neurons, 0.6f,
                         imagination_system->memory_influence);
      simulateScenario(&problem_scenario, neurons, input_tensor, max_neurons,
                       15, weights, connections, max_connections);

      // Blend all outcomes for a comprehensive solution
      float blended_solution[MEMORY_VECTOR_SIZE] = {0};
      blendImaginedOutcomes(problem_scenario.outcomes,
                            problem_scenario.num_outcomes, blended_solution);

      // Apply blended solution with stronger influence during difficult
      // problems
      for (int i = 0; i < max_neurons && i < MEMORY_VECTOR_SIZE; i++) {
        neurons[i].state = neurons[i].state * 0.7f + blended_solution[i] * 0.3f;
        input_tensor[i] = input_tensor[i] * 0.8f + blended_solution[i] * 0.2f;
      }

      printf("Applied blended imagination solution to difficult problem\n");
    }

    if (step % 30 == 0 && imagination_system->num_scenarios > 0) {
      // Find most successful scenario (highest plausibility × confidence)
      int best_idx = 0;
      float best_score = 0.0f;

      for (int i = 0; i < imagination_system->num_scenarios; i++) {
        float score =
            imagination_system->scenarios[i].outcomes[0].plausibility *
            imagination_system->scenarios[i].outcomes[0].confidence;
        if (score > best_score) {
          best_score = score;
          best_idx = i;
        }
      }

      // Store in memory system
      MemoryEntry new_memory;
      memcpy(new_memory.vector,
             imagination_system->scenarios[best_idx].outcomes[0].vector,
             MEMORY_VECTOR_SIZE * sizeof(float));
      new_memory.importance = best_score;
      new_memory.timestamp = lastTimestamp + step;

      // Add to memory system
      addToDirectMemory(memorySystem, &new_memory);
      printf("Stored successful imagination scenario in memory\n");
    }

    if (step % 10 == 0) {
      consolidateToLongTermMemory(working_memory, memorySystem, step);
    }
    updateBidirectionalWeights(weights, reverse_weights, neurons, connections,
                               reverse_connections, learning_rate);

    float decision_vector[5] = {0}; // One value per ethical principle

    // Map network state to ethical dimensions
    for (int i = 0; i < 5 && i < max_neurons / 10; i++) {
      for (int j = 0; j < 10 && i * 10 + j < max_neurons; j++) {
        decision_vector[i] += neurons[i * 10 + j].output * 0.1f;
      }
      decision_vector[i] = fmax(0.0f, fmin(1.0f, decision_vector[i]));
    }

    float average_error = total_error / max_neurons;
    if (step % 15 == 0) {
      advancedNeuronManagement(neurons, connections, weights, &max_neurons,
                               MAX_NEURONS, input_tensor, target_outputs,
                               stateHistory, step);
    }

    if (step % 5 == 0) {
      float affective_satisfaction =
          (1.0f - average_error) * (1.0f + aff_sys->current_state.valence);

      detectEmotionalTriggers(emotional_system, updatedNeurons, target_outputs,
                              max_neurons, lastTimestamp + step + 1,
                              affective_satisfaction, aff_sys, social_system);
    }

    applyEmotionalProcessing(emotional_system, updatedNeurons, max_neurons,
                             input_tensor, learning_rate, params.plasticity,
                             aff_sys);

    if (step % 10 == 0 && aff_sys != NULL) {
      integrateAttachmentsIntoIdentity(aff_sys, identity_system->core_values,
                                       identity_system->num_core_values);

      if (step % 30 == 0) {
        printf("\nIdentity values after integration:\n");
        for (uint32_t i = 0; i < identity_system->num_core_values; i++) {
          printf("  Value[%u]: %.3f\n", i, identity_system->core_values[i]);
        }
      }
    }

    if (step % 20 == 0) {
      printEmotionalState(emotional_system);
      simulateEmotionalTrajectory(aff_sys, social_system,
                                  feedback.context_weights, step + 1);
    }

    if (step % 100 == 0 && step > 0) {
      printAttractorAnalysis(aff_sys);
    }

    // Adjust emotional regulation based on performance
    if (step % 20 == 0) {
      // Increase regulation as the system learns
      emotional_system->emotional_regulation =
          fmin(0.9f, emotional_system->emotional_regulation + 0.01f);

      // Slowly increase cognitive impact to allow more emotional influence
      emotional_system->cognitive_impact =
          fmin(0.5f, emotional_system->cognitive_impact + 0.005f);
    }

    for (int i = 0; i < 5; i++) {
      recordDecisionOutcome(moralCompass, i, decision_vector[i] >= 0.7f);
    }

    systemFallbackCheck(
        neurons, (int *)connections, weights, (int *)reverse_connections,
        reverse_weights, memorySystem, stateHistory, performance_history,
        input_tensor, target_outputs, previous_outputs, system_params,
        working_memory, metaController, performanceMetrics, motivation,
        reflection_params, identity_system, knowledge_filter, metacognition,
        meta_learning_state, social_system, goalSystem, contextManager,
        emotional_system, imagination_system, specialization_system,
        moralCompass, step, max_neurons, max_connections, input_size);

    NSLog(@"Average Error: %f", average_error);
    double throughput = STEPS / performance_history[step].execution_time;
    NSLog(@"Throughput: %f steps/s", throughput);
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    printf("Memory Usage Benchmark:\n");
    printf("Max Resident Set Size: %ld KB\n", usage.ru_maxrss);
  }

  // Save final state
  saveNetworkStates(stateHistory, STEPS);
  saveMemorySystem(memorySystem, "memory_system.dat");
  saveHierarchicalMemory(memorySystem, "hierarchical_memory.dat");
  saveSystemParameters(system_params, "system_parameters.dat");
  saveAllSystems(metaController, motivation, performanceMetrics,
                 reflection_params, identity_system, knowledge_filter,
                 metacognition, meta_learning_state, social_system);

  printf("\nNeural network states, memory system and system parameters have "
         "been saved\n");

  generatePerformanceGraph(performance_history, STEPS);

  // Cleanup
  freeDatasetLoader(dataset);
  freeWorkingMemorySystem(working_memory);
  freeMemorySystem(memorySystem);
  freeMoralCompass(moralCompass);
  freeEmotionalSystem(emotional_system);
  freeImaginationSystem(imagination_system);
  freeSocialSystem(social_system);
  freeAffectiveSystem(aff_sys);
  freeGlobalContextManager(contextManager);
  freeKnowledgeFilter(knowledge_filter);
  freeGoalSystem(goalSystem);
  freeSelfIdentitySystem(identity_system);
  cleanupEmbeddings();
  free(input_tensor);
  free(stateHistory);
  free(system_params);
  free(performance_history);
  free(performanceMetrics);
  free(metaController);
  free(previous_outputs);
  free(motivation);
  free(reflection_params);
  free(metacognition);
  free(meta_learning_state);
  free(specialization_system);
  return 0;
}
