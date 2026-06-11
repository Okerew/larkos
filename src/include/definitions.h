#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#define MAX_NEURONS 8
#define MAX_CONNECTIONS 6
#define STEPS 100
#define INPUT_SIZE 6            // Size of the input tensor
#define MEMORY_BUFFER_SIZE 1000 // Size of circular memory buffer
#define MEMORY_VECTOR_SIZE (2 * MAX_NEURONS + INPUT_SIZE)
#define DECAY_FACTOR 0.95f           // Decay factor for memory over time
#define CONSOLIDATION_THRESHOLD 0.7f // Threshold to consolidate memories
#define STRENGTHEN_FACTOR 1.2f       // Factor to increase memory importance
#define REMOVE_THRESHOLD 0.05f // Threshold below which memory is forgotten
#define OPTIMIZATION_WINDOW 5  // Number of steps to consider for optimization
#define PERFORMANCE_THRESHOLD 0.8 // Target performance improvement threshold
#define MAX_BATCH_SIZE 16         // Maximum batch size for processing
#define EMBEDDING_SIZE 16         // Size of word embeddings
#define WEIGHT_DECAY 0.95f        // Weight decay factor
#define MAX_SIMULATIONS 10        // Number of simulation runs
#define DECAY_RATE 0.8f
#define INPUT_WEIGHT 0.1f
#define CONNECTION_WEIGHT 0.2f
#define ACTIVATION_SCALE 1.5f
#define ACTIVATION_BIAS 0.1f
#define MIN_ACTIVATION -1.0f
#define MAX_ACTIVATION 1.0f
#define LEARNING_RATE 0.01f
#define MIN_WEIGHT -1.0f
#define MAX_WEIGHT 1.0f
#define MAX_SIMULATIONS 10 // Number of simulation runs
#define NUM_TIME_STEPS 20
#define FEATURE_VECTOR_SIZE 128
#define CONTEXT_VECTOR_SIZE 256
#define CLAMP_MIN -1e6f // Min value for feature or coherence
#define CLAMP_MAX 1e6f  // Max value for feature or coherence
#define PATTERN_SIZE 3
#define EXPERIENCE_VECTOR_SIZE 256
#define HISTORY_LENGTH 10
#define NUM_PATHS 5
#define MAX_DECISION_STEPS 20
#define MAX_USAGE_COUNT 1000 // Maximum usage count for normalization
#define MAX_SYMBOLS 100
#define MAX_QUESTIONS 10
#define VOCAB_SIZE 100
#define ACTIVATION_TANH 0
#define ACTIVATION_RELU 1
#define ACTIVATION_SIGMOID 2
#define ACTIVATION_LEAKY_RELU 3
#define ACTIVATION_SWISH 4
#define MAX_EMOTION_TYPES 8
#define EMOTION_LOVE 0
#define EMOTION_HATE 1
#define MAX_SCENARIOS 10
#define MAX_SCENARIO_STEPS 20
#define MAX_SCENARIO_NAME_LENGTH 100
#define MAX_SPECIALIZATIONS 8
#define MAX_SPECIALIZED_NEURONS 64
#define MAX_OUTCOMES_PER_SCENARIO 10
#define SPARSE_DENSITY                                                         \
  0.05f // Only 5% of dimensions active (like cortical columns)
#define NUM_SEMANTIC_LAYERS 4 // Hierarchical representation layers
#define CONTEXT_WINDOW 8      // Context for dynamic embeddings
#define HASH_BUCKETS 1024     // For efficient similarity SearchResults
#define HARM_WEIGHT 0.5f
#define UNCERTAINTY_WEIGHT 0.3f
#define BENEFIT_WEIGHT 0.2f
#define MAX_TEXT_LENGTH 4096
#define MAX_TOKENS 512
#define MAX_TOKEN_LENGTH 64
#define NGRAM_SIZE 3
#define NUM_HEADS 8
#define HEAD_DIM (EMBEDDING_SIZE / NUM_HEADS)
#define DROPOUT_RATE 0.1f
#define MAX_LINE_LENGTH 10000
#define MAX_WORD_LENGTH 100
#define PREDICTION_WINDOW 10
#define PREDICTION_HISTORY_WEIGHT 0.85f
#define PREDICTION_CURRENT_WEIGHT 0.15f
#define MIN_PREDICTION_SAMPLES 5
#define PREDICTION_ERROR_DECAY 0.95f
#define TEMPORAL_PREDICTION_STEPS 3
#define MAX_SAMPLES 100000
#define MAX_EMOTION_ATTRACTORS 20
#define MAX_ATTACHMENT_BONDS 50
#define EMOTION_HISTORY_SIZE 100
#define MAX_EMOTION_PATTERNS 10
#define NEURON_STRIDE 4
#define ACTIVATION_HISTORY_SIZE 50
#define HISTORY_SIZE 100
#define REASONING_SIZE 1024

typedef struct {
  float state;
  float output;
  unsigned int num_connections;
  unsigned int layer_id;
} Neuron;

#endif // DEFINITIONS_H
