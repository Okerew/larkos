# Larkos Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Key Components](#key-components)
3. [Key Functions](#key-functions)
4. [An idea of a neural web](https://github.com/Okerew/larkos/blob/main/Documents/an_idea_of_a_neural_web.pdf)
5. [Neural Web training mechanism](https://github.com/Okerew/larkos/blob/main/Documents/training_example.md)

## Introduction

<img src="structure.png" alt_text="watch the video">

This documentation provides a comprehensive guide to the Lattice-Associated Recursive Knowledge Organization System (or decentralized neural web architecture) implemented in the provided code. The architecture is designed to simulate a neural network with hierarchical memory management, dynamic adaptation, and performance optimization along with other features.
The goal of this architecture is to present an alternative to modern neural models, which are often complex and resource-intensive, taking inspiration from our brains, neurons are decentralized organized in layers, allowing them to interact with themselves and change themselves overtime in more than just sates and weights, while also creating a dynamic memory system.

<a href="https://mermaid.live/edit#pako:eNqNWGtv2zgW_SuEgAk6QDLbOq_WAyzgxE7iady4kZtiZzIfaIm2OZFIg6LSeur-9z2XokTJznTXH_wQ7_tx7qW_RYlORdSPloavV2w2_PVRMbwGfzxGY7UuLRtyyx-jP9nR0b-3U6MTURQiZfPNll2A5oMojVYFCDzjhSP8tE65FSy2eN-yy4awesKq412mS51DoWB3pcVnsWVD8A0SK5-5leC9KlXivryyXK1-DuyXbZ2fhVyuiHsE7uqH18de3Yj5XHLFbgU3SqplS8bIyRikf5WFDTIu_OnQnV4LJQwJmohcmw17EInVZsuuoMk_mnDFlyIXygbR1ftPP9Vs8aawIq-eXjnBMcR4qVJA7TUE3kjoMslKJjzznEHktWO7Fxb0z4KN87U2CIptybgJRnk6ngUBN07ARJglsiRzmXHT4h23HAIJIrXL-akQbKENmxqRSpeVLfsNXPVvGHWp0w5j5epQJHzDDnCqCp3JlFe874PG2plkv07eOxF3awuL_27SQNFD0BE2kDXB_qzNE_TXRGNlBWqctLXNmQnSxttSPk-ocDrcwYLPk6pOikKiTprAItxdgrhckxdsUMXCNw4khprylBdGppSEFZEfsFutlkczYXKvuvKqU0bXmZ6jJhBBK75aX3KGfSrw0Y7TICGdNR0EUR--zNyqrKqZxmqRlYJysJcVJKtLC-eeZSoamVYzDwuVs43h92KRiaqF2z0wrUNK1lbNzz6WPJMW3t_HMHqfsbH3Pm6wg6OHoHv0dQ1aYFQlCkDwABEjZKms0h96_qHyNUW7ysWGjYzRpkAS4Cx8gqdX1AlXQqRznjzRgRVmR8jVuI0cKG9ZkJ0-4yHdng6uSCUaBELZEaIs4elkN8-xyBZH3jigwF7cKtQcKJ5t0AxU4Abfa9CNyfQfiGgciMe-MS3Cxi4kRx7IVVVYPpfIgoOEmNCeDnEU-4MNu1yJ5Kkl6qIdi92ibxwLppJ5yJchssY09mocj-KfO07OC8QJ5r1X-ksmqGEO2JRbkkN4BXo3sH4oN9hJ9FWbbvI5UChBaRIx4uTxKP4PoQCSUya2BKax2UqXNEpuuVqW1Gj7wjwovBBzysYu9UitHMaF2qZi36XylRjqasKfXECnu5S-sHwTon3YJ5UKgySq1HFcX7ZzMNHNXG3XlG_poSHUaor0wOEYWTihOO_zNsGYxB4SJEaJbYE0RCBbqpJy3aX29dKou3cFTAkYbhTPkZ5Bytd2p_E8c0AqDx11qIoqSI3Ho1z_UwstFlT6oV59FwA6JneEHV3OgB-Tu2qM6rTMaC9oXKTqrEJAjR3v0Nepx_ykOUFV2h7-O9TBwZGttoF7wQutXFIno31qjeHRAv0dgpmRSxoXLqSU5XtRrBEtUYSi8qDq_YbGGgJdRNpBbXKGKmN-MLeG7G8df2viLbulVUEYMtWddDib-N6297opNzwHRpkiQGU1QWdlG1CpeA4DRG3ZB-h6Wf6Hnfj6HfWlYX3rE62kpawF6Xdut7VfsC-0cHGCDAE2unvgXaejQ0lX_jQhDRB3JTP42ynYgZPhDw7YA3crlHBbOgYzbaK7_K316cqng2wLI_bhI7gehAkj4qA-dHVkdGtxfPjo0SYnJ8bGiAwLkLLegkGXrHb2f6BSZ-w1UHfQlGa7WZslvB7vUxrRkqq05gzWTusFJS2TUOlbRh7PaFfxmPEvD3AVZwskDcRSadetd11i08kAtEVbgV8uQofCb0wrak8HlyTloDmk1RdSqkWidXWajGrsVIlcZ4K96bOhZkqzFTc5HL15Q5DYPOmzwbOWKePO9ILZFbcs4SVW81IpQTVMi21RAt4MtQYqN-V5Z3jt6ez1XZgID9cYAjzZOP95abXS-Yas6LmN7J9pmF4wnmWMoJCWhx9oO-6zCyxtprSrRVlFmidJadwAmN4cU_S4xFSWqqFSBJeBEspxSPr0-v-I6UmfXXFpGiFKq6NUFokBPijfjtObE0L9qk0WbWruAt6lr9Wv_eW42CmiWCeSdpJWF4dBwOewaYRqsSvENqbp2qUPu1W96GqTkuYargr2RdpVreVa86y1dHqm-jYC5ipGO1eSpgnj7m5SV2yAfzdvwiKHYqqjsD9Wm0692ay1XQlLsmhbo8n-AmvYqPwlq7r6C_YgjYOOzq2iRVhdnyuJWNVGz6i81iD1VFho1xJ0l0bsR4DWqE7SUN2Ipx8ZLzg3M3QjmPHiqbp4IXkERC_zhSQ2dx25VEEJjL7XmWgnzhPSkGB3zwD7mcxFmHz-3C99fnDVf5BAzodfo8MoxzWSyzTqR9-I6zFCEnI0fx9fU7HgZYbp9Ki-g5R6N96oJOqjzcRhZGjbjfoLJAy_SjeCh5LjBp03T9dc_a51XrPgZ9T_Fn2N-r3TX96en7x7e3bWOz8_P319chht8PT4-JeT3uuTXu_47fnZm3cnp98Po7-dgNegP32NV-_k7Pis9-60dxgtDVnurRE0NC51qWzUPz7GqUhpEE-q_67cX1jf_wumlDOh">You can see the architecture better here</a>

**Video explanations:**
<a href="https://www.youtube.com/watch?v=dw4DhJ3TPBY">First part</a>
<a href="https://www.youtube.com/watch?v=c0p1_2vjNic">Second part</a>

## Requirements:

gnuplot library

json-c library

for metal macos version metal api

for cuda version cuda and pybind11

golang(for converter and train embeddings)

python(for running cuda version and for running libary file through ctypes)

docker(optional)

If on windows use wsl i have removed the windows version as I don't think it was ever correctly working either way.

## Larkos 0.1

Introducing Larkos 0.1 a self-learning model with a state-based fusion mechanism, transformers that guide the model, based on the larkos architecture.

<a href="https://github.com/Okerew/larkos_0.1">Visit github page</a>

## Disclaimer

<img src="https://github.com/Okerew/Neural-Web/blob/main/security_logs.png">

It's recommended you use this more in a framework kind of way than by just copying the whole int main functionality basicly use functions you like from here compilation guide is below.

It's good practice to use the validating critical security status functions if doing more complex things with the code.

```c

SecurityValidationStatus secStatus =
        validateCriticalSecurity(updatedNeurons, weights, connections,
                                 max_neurons, max_connections);

if (secStatus.critical_violation) {
      handleCriticalSecurityViolation(updatedNeurons, weights, connections, &secStatus);
}
```

## Library Installation 
You can either setup the C version and use it with ctypes by building it as a libary with unix 64x support or you can compile the neural web on macOS if you really want but I wouldn't recommend that.
## Building the Neural Web

### Recommended way (build with docker or pull from docker hub)

Find the correct version you want to build by downloading the whole repo `git clone https://github.com/Okerew/Neural-Web.git` and navigating to the correct version you want to build.

#### Firstly

If you seriously want to do it like I did start by firstly generating embeddings with train_embedding main file which you can compile like this `go build`, then run with `./main` this should generate a embeddings file (custom_embeddings.txt) if you didn't change the name which then copy to the directory were you will be building the neural web.

### Compilation

#### Firstly
Start by firstly generating embeddings with train_embeddings main file which you can compile like this `go build`, then run with `./main` this should generate a embeddings file (custom_embeddings.txt) if you didn't change the name which then copy to the directory were you will be building the neural web, note although this isn't needed it is needed if you want to fully use larkos.

To compile the code, run the following command in the root directory of the project:
#### C through Ctypes
```sh
gcc -shared -fPIC -o neural_web.so \
neural_web.c immitrin_functions.c \
-ljson-c
```

To use (You can see more in the test_ctypes.py example)
```python
from ctypes import CDLL
import os

lib_path = os.path.join(os.getcwd(), "neural_web.so")
lib = CDLL(lib_path)
```
#### arch64 MacOS

```sh
# Build executable
clang MacOS.m neural_web.c simd_functions.c \
-framework Metal -framework Foundation \
-I/opt/homebrew/Cellar/json-c/0.18/include \
-L/opt/homebrew/Cellar/json-c/0.18/lib -ljson-c \
-O1 -o neural_web

# Build dynamic library
clang -dynamiclib \
neural_web.c simd_functions.c \
-I/opt/homebrew/Cellar/json-c/0.18/include \
-L/opt/homebrew/Cellar/json-c/0.18/lib -ljson-c \
-O1 \
-o libneural_web.dylib

```

JsonC library replace with your own imports in the command if you copied it into the project or aren't using homebrew or another version of the lib

1. Note: the actual code is located in src so you should either unpack files from there into your folder or just compile there
2. Note: you can use the install.sh script to be able to use the lib straight from your os.

## Architecture Overview

The architecture consists of several key components:

- **Neurons**: The basic units of the neural network, organized into layers, each connected in a 3d like structure.
  ![alt text](neuron_connections_3d.png)
- **Memory System**: A hierarchical memory system to store and manage memories with varying importance.
- **Dynamic Parameters**: Parameters that adapt based on the network's performance and stability.
- **Performance Metrics**: Metrics to track the performance of the network.
- **Optimization**: Techniques to optimize the network's parameters for better performance.
  ![alt text](image.png)
- **Reflection System**: Evaluates the quality of outputs and suggests improvements.
- **Self-Identification System**: Helps the system assess its own state and biases, allowing the AI to form an identity of sorts.
- **Knowledge Filter**: Ensures that only relevant and high-quality information is processed.

## Loss figure :

![alt text](image-1.png)

## Key Components

### Memory System

<img src="https://github.com/Okerew/Neural-Web/blob/main/memory_system.png">

The memory system is designed to store and manage memories with varying importance. It consists of:

- **MemoryEntry**: A structure to store individual memories.
- **MemoryCluster**: A structure to manage a cluster of memories.
- **HierarchicalMemory**: A structure to manage short-term, medium-term, and long-term memories.
- **MemorySystem**: The main structure to manage the hierarchical memory system.

#### MemoryEntry

```c
typedef struct {
  float vector[MEMORY_VECTOR_SIZE];
  float importance;
  unsigned int timestamp;
} MemoryEntry;
```

#### MemoryCluster

```c
typedef struct MemoryCluster {
  MemoryEntry *entries;
  float importance_threshold;
  unsigned int size;
  unsigned int capacity;
} MemoryCluster;
```

#### HierarchicalMemory

```c
typedef struct HierarchicalMemory {
  MemoryCluster short_term;
  MemoryCluster medium_term;
  MemoryCluster long_term;
  float consolidation_threshold;
  float abstraction_threshold;
  unsigned int total_capacity;
} HierarchicalMemory;
```

#### MemorySystem

```c
typedef struct MemorySystem {
  HierarchicalMemory hierarchy;
  unsigned int head;
  unsigned int size;
  unsigned int capacity;
  MemoryEntry *entries;
} MemorySystem;
```

### Neuron, Connection Management, neuron specialization

Neurons are the basic units of the neural network, organized into layers, each connected in a 3d like structure and specialized.

#### Neuron

```c
typedef struct {
  float state;
  float output;
  unsigned int num_connections;
  unsigned int layer_id;
} Neuron;

typedef enum {
  SPEC_NONE = 0,
  SPEC_PATTERN_DETECTOR,
  SPEC_FEATURE_EXTRACTOR,
  SPEC_TEMPORAL_PROCESSOR,
  SPEC_CONTEXT_INTEGRATOR,
  SPEC_DECISION_MAKER,
  SPEC_MEMORY_ENCODER,
  SPEC_EMOTIONAL_PROCESSOR,
  SPEC_PREDICTION_GENERATOR
} NeuronSpecializationType;

typedef struct {
  unsigned int neuron_id;
  NeuronSpecializationType type;
  float specialization_score;
  float activation_history[50]; // Recent activation history
  unsigned int history_index;   // Current index in circular buffer
  float avg_activation;         // Average activation level
  float importance_factor;      // How important this specialized neuron is
} SpecializedNeuron;

typedef struct {
  SpecializedNeuron neurons[MAX_SPECIALIZED_NEURONS];
  unsigned int count;
  float type_distribution[MAX_SPECIALIZATIONS]; // Distribution of
                                                // specialization types
  float specialization_threshold; // Minimum score to be considered specialized
} NeuronSpecializationSystem;
```

### Dynamic Parameters

Dynamic parameters adapt based on the network's performance and stability.

#### DynamicParameters

```c
typedef struct {
  float input_noise_scale;
  float weight_noise_scale;
  float base_adaptation_rate;
  float current_adaptation_rate;
  float learning_momentum;
  float stability_threshold;
  float noise_tolerance;
  float recovery_rate;
  float plasticity;
  float homeostatic_factor;
} DynamicParameters;
```

### Performance Metrics

Performance metrics track the performance of the network.

#### PerformanceMetrics

```c
typedef struct {
  double execution_time;
  float average_output;
  float error_rate;
  int batch_size;
  float learning_rate;
} PerformanceMetrics;
```

### Optimization

Optimization techniques are used to improve the network's performance.

#### OptimizationState

```c
typedef struct {
  int optimal_batch_size;
  float optimal_learning_rate;
  double best_execution_time;
  float best_performance_score;
} OptimizationState;
```

### Reflection System

```c
typedef struct {
  float current_adaptation_rate;
  float input_noise_scale;
  float weight_noise_scale;
  float plasticity;
  float noise_tolerance;
  float learning_rate;
} ReflectionParameters;
```

The reflection system evaluates the quality of outputs and suggests improvements. It helps in continuously refining the network's performance by identifying areas that need enhancement.

### Self-Identification System

```c
typedef struct {
  float *core_values;         // Stable personality traits/values
  float *belief_system;       // Current belief states
  float *identity_markers;    // Unique identifying characteristics
  float *experience_history;  // Compressed history of experiences
  float *behavioral_patterns; // Consistent behavior patterns

  uint32_t num_core_values;
  uint32_t num_beliefs;
  uint32_t num_markers;
  uint32_t history_size;
  uint32_t pattern_size;

  float consistency_score; // Measure of identity stability
  float adaptation_rate;   // Rate of identity evolution
  float confidence_level;  // Self-confidence in identity

  // Temporal consistency tracking
  float *temporal_coherence; // Track consistency over time
  uint32_t coherence_window; // Time window for coherence analysis

  // Identity verification system
  struct {
    float threshold;        // Minimum consistency threshold
    float *reference_state; // Reference identity state
    uint32_t state_size;    // Size of reference state
  } verification;

} SelfIdentitySystem;
```

The self-identification system helps the neural web assess its own state and biases. This allows the AI to form an identity of sorts, enabling it to understand its capabilities and limitations better.

### Knowledge Filter

The knowledge filter ensures that only relevant and high-quality information is processed. This component is crucial for maintaining the integrity and efficiency of the neural web by filtering out noise and irrelevant data.

```c
typedef struct {
  KnowledgeCategory *categories;
  uint32_t num_categories;
  uint32_t capacity;
  ProblemInstance *problem_history;
  uint32_t num_problems;
  uint32_t problem_capacity;
  float *category_similarity_matrix;
} KnowledgeFilter;
```

### Metacognition

The metacognition system evaluates the performance of the neural web and suggests improvements. It helps in continuously refining the network's performance by identifying areas that need enhancement.

```c
typedef struct MetacognitionMetrics {
  float confidence_level;                    // Overall confidence in decisions
  float adaptation_rate;                     // Rate of learning adjustment
  float cognitive_load;                      // Current processing complexity
  float error_awareness;                     // Awareness of prediction errors
  float context_relevance;                   // Relevance of current context
  float performance_history[HISTORY_LENGTH]; // Historical performance tracking
} MetacognitionMetrics;

typedef struct MetaLearningState {
  float learning_efficiency; // Current learning effectiveness
  float exploration_rate;    // Balance between exploration/exploitation
  float stability_index;     // System stability measure
  float *priority_weights;   // Attention allocation weights
  uint32_t current_phase;    // Current learning phase
} MetaLearningState;
```

### Security system 

The security system evaluates it the network is trying to access the system. It helps in preventing unauthorized access to the system.

```c
typedef struct {
  bool critical_violation;
  uint64_t suspect_address;
  const char *violation_type;
} SecurityValidationStatus;
```

### Goal system

The goal system allows the network to set and track goals.

```c
typedef struct {
  float novelty_score;
  float competence_score;
  float autonomy_score;
  float mastery_level;
  float curiosity_drive;
  float achievement_drive;
  float exploration_rate;
} IntrinsicMotivation;

typedef struct {
  char description[256];
  float priority;
  float progress;
  float reward_value;
  bool achieved;
  int timestamp;
} Goal;

typedef struct {
  Goal *goals;
  int num_goals;
  int capacity;
  float planning_horizon;
  float discount_factor;
} GoalSystem;
```

### Internal self expression system

The internal self expression system allows the network to express itself. It allows the network to ask questions about it self and get answers

```c
typedef struct {
    int symbol_id;
    char description[256];
} InternalSymbol;

typedef struct {
    int question_id;
    int symbol_ids[MAX_SYMBOLS];
    int num_symbols;
} InternalQuestion;
```

### Moral compass

The moral compass ensures the model adheres to basic ethical principles. It allows the model to make decisions that are aligned with ethical standards.

```c
typedef struct {
  char **titles;
  char **snippets;
  char **urls;
  int count;
} SearchResults;

typedef struct {
  float importance;      // How important this principle is (0.0-1.0)
  float adherence;       // Current adherence level (0.0-1.0)
  char description[256]; // Description of the principle
  int violations;        // Count of violations
  int activations;       // Count of successful applications
} EthicalPrinciple;

typedef struct {
  float benefit_score;    // Positive impact measurement
  float harm_score;       // Negative impact measurement
  float uncertainty;      // Level of uncertainty in assessment
  int affected_parties;   // Number of parties potentially affected
  float reversibility;    // How reversible the decision is (0-1)
  float long_term_impact; // Long-term consequence rating
} DecisionImpact;

typedef struct {
  EthicalPrinciple *principles; // Array of ethical principles
  int num_principles;           // Number of principles
  float overall_alignment;      // Overall ethical alignment (0.0-1.0)
  DecisionImpact last_decision; // Impact of the last decision
  float confidence_threshold;   // Minimum confidence for ethical decisions
  int dilemma_count;            // Number of ethical dilemmas encountered
  int resolution_count;         // Number of dilemmas successfully resolved
} MoralCompass;
```

### Affective system

The affective system servers as en extension of the emotion system allowing for more complex emotional simulation.

```c
typedef struct {
  uint32_t entity_id;
  char entity_name[64];
  float attachment_strength;
  float trust;
  float familiarity;
  float dependency;
  float care_investment;
  float loss_cost;
  uint32_t shared_history_index;
  float emotional_resonance;
  float conflict_history;
  uint32_t interaction_count;
  float last_interaction_valence;
  float predicted_behavior_alignment;
  float emotional_debt;
} AttachmentBond;

typedef struct {
  float valence;
  float arousal;
  float dominance;
  float complexity;
  float temporal_depth;
  uint32_t duration_steps;
  float stability;
  float momentum[3];
} EmotionVector;

typedef struct {
  uint32_t attractor_id;
  char attractor_name[64];
  EmotionVector center_point;
  float basin_strength;
  float entry_threshold;
  float exit_threshold;
  float stability_factor;
  uint32_t visit_count;
  float average_duration;
  float *context_weights;
  uint32_t context_dim;
  bool is_pathological;
  float reinforcement_rate;
  uint32_t linked_attractors[5];
  float transition_probabilities[5];
} EmotionAttractor;

typedef struct {
  EmotionVector current_state;
  EmotionVector base_state;
  EmotionAttractor *attractors;
  uint32_t num_attractors;
  uint32_t current_attractor_id;
  uint32_t steps_in_current_attractor;
  EmotionVector history[EMOTION_HISTORY_SIZE];
  uint32_t history_index;
  float *affective_embeddings;
  uint32_t embedding_dim;
  float plasticity;
  float self_complexity;
  AttachmentBond *bonds;
  uint32_t num_bonds;
  uint32_t max_bonds;
  float relational_bias;
  float predictive_commitment_weight;
  float subconscious_influence;
} AffectiveSystem;
```

### Emotion system

The emotion system allows the network to express emotions. It allows the network to express emotions and get answers

```c
typedef struct {
  float intensity;          // Strength of the emotion (0.0 to 1.0)
  float decay_rate;         // How quickly the emotion fades
  float influence_factor;   // How much this emotion affects decision making
  float threshold;          // Activation threshold for this emotion
  float previous_intensity; // For tracking changes
  float momentum;           // Carries emotional momentum across steps
  unsigned int last_update; // Timestamp of last update
} EmotionState;

typedef struct {
  EmotionState emotions[MAX_EMOTION_TYPES];
  float cognitive_impact;     // How much emotions affect logical processing
  float emotional_regulation; // System's ability to regulate emotions (0.0-1.0)
  float emotional_memory[MAX_EMOTION_TYPES]
                        [10]; // Recent emotional memory traces
  int memory_index;           // Current index in circular memory buffer
} EmotionalSystem;
```

### Imagination System

The imagination system allows the network to generate imaginative outcomes. It allows the network to generate imaginative outcomes and get answers

```c
typedef struct {
  float probability;
  float confidence;
  float impact_score;
  float plausibility;
  float vector[MEMORY_VECTOR_SIZE];
  char description[256];
} ImaginedOutcome;

typedef struct {
  int num_outcomes;
  ImaginedOutcome outcomes[10];
  float divergence_factor;
  float creativity_level;
} ImaginationScenario;

typedef struct {
  ImaginationScenario scenarios[MAX_SCENARIOS];
  int num_scenarios;
  int current_scenario;
  float creativity_factor;
  float coherence_threshold;
  float novelty_weight;
  float memory_influence;
  float identity_influence;
  bool active;
  int steps_simulated;
  float divergence_history[100];
  char current_scenario_name[MAX_SCENARIO_NAME_LENGTH];
  int total_scenarios_generated;
} ImaginationSystem;
```

### Social system

The social system allows the network to interact with others. It allows the network to interact with others and get answers

```c
typedef struct {
  unsigned int timestamp;
  int person_id;              // ID of the person involved
  float emotional_state[5];   // Emotional state during interaction
  float cooperation_level;    // How cooperative the interaction was
  float outcome_satisfaction; // How satisfied both parties were
  char interaction_type[32];  // Type of interaction (negotiation, casual, etc.)
  char *context;              // Context of the interaction
} SocialInteraction;

// Structure to model another person
typedef struct {
  int person_id;
  char person_name[64];
  float observed_traits[10];   // Personality traits inferred
  float prediction_confidence; // Confidence in behavioral predictions
  float relationship_quality;  // Quality of relationship with this person
  float trust_level;           // Trust built with this person
  int interaction_count;       // Number of interactions with this person
} PersonModel;

typedef struct {
  // Core social capabilities
  float empathy_level;     // Ability to understand others' emotions
  float negotiation_skill; // Ability to find mutually beneficial solutions
  float behavior_prediction_accuracy; // Accuracy in predicting others' actions
  float social_awareness;             // Awareness of social dynamics and norms

  // Social interaction history
  int interaction_count;
  SocialInteraction *interactions; // Array of past interactions
  int max_interactions;            // Maximum number of interactions to store

  // Social models of others
  int model_count;
  PersonModel
      *person_models; // Models of individuals the system has interacted with
  int max_models;     // Maximum number of models to maintain

  // Social learning parameters
  float learning_rate;     // Rate at which social skills improve
  float forgetting_factor; // Rate at which old interactions lose relevance
} SocialSystem;
```

### Validation system

The validation system allows for the neural web to fallback if encountering an error.

```c
typedef struct {
  time_t start_time;
  unsigned long total_checks;
  unsigned long successful_checks;
  unsigned long failed_checks;
  unsigned long segfaults_recovered;
  unsigned long fpe_recovered;
  float average_check_time;
  float min_check_time;
  float max_check_time;
  float total_check_time;
  unsigned long component_failures;
  unsigned long memory_issues;
  unsigned long instability_events;
  unsigned long critical_failures;
  unsigned long neuron_corrections;
  unsigned long connection_corrections;
  unsigned long weight_corrections;
  unsigned long memory_reinitializations;
  unsigned long memory_cluster_errors;
} SystemHealthMetrics;
```

# Key Functions:

## Memory System

- #### `createMemorySystem(int capacity)` Initializes a new memory system with a specified capacity. Sets up a data structure to hold a certain number of memory entries.
- #### `loadMemorySystem(const char* filename)` Loads a memory system from a file. Reads a file to populate the memory system with previously saved data.
- #### `saveMemorySystem(MemorySystem memorySystem, const char* filename)` Saves the current memory system to a file. Writes the current state of the memory system to a file for later use.
- #### `freeMemorySystem(MemorySystem memorySystem)` Frees the memory allocated for the memory system. Deallocates memory to prevent memory leaks.
- #### `loadHierarchicalMemory(MemorySystem memorySystem, const char* filename)` Loads hierarchical memory from a file. Involves loading memory data that has a hierarchical structure, such as categories and subcategories.
- #### `saveHierarchicalMemory(MemorySystem memorySystem, const char* filename)` Saves hierarchical memory to a file. Saves the hierarchical structure of the memory system to a file.
- #### `decayMemorySystem(MemorySystem memorySystem)` Applies decay to the memory system to simulate forgetting. Reduces the strength or importance of older memories.
- #### `mergeSimilarMemories(MemorySystem memorySystem)` Merges similar memories to optimize storage. Combines memories that are very similar to save space.
- #### `addMemory(MemorySystem memorySystem, WorkingMemorySystem working_memory, Neuron* neurons, float* input_tensor, int timestamp, float feature_projection_matrix[FEATURE_VECTOR_SIZE][MEMORY_VECTOR_SIZE])` Adds a new memory entry to the system. Takes input data and stores it as a new memory.
- #### `retrieveMemory(MemorySystem memorySystem)` Retrieves the most relevant memory entry. Searches the memory system for the memory that best matches a certain criterion.
- #### `consolidateMemory(MemorySystem memorySystem)` Consolidates memories to reinforce learning. Strengthens important memories or transfers them to long-term storage.
- #### `consolidateToLongTermMemory(WorkingMemorySystem working_memory, MemorySystem memorySystem, int step)` Consolidates working memory to long-term memory. Moves short-term memories to long-term storage.
- #### `addToDirectMemory(MemorySystem *memorySystem, const MemoryEntry *entry)` Adds a memory entry directly to the memory system. Adds the memory entry to the memory system without any integration with the working memory.

## Neural Network

- #### `initializeNeurons(Neuron* neurons, int connections, float* weights, float* input_tensor)` Initializes the neurons with default or loaded values. Sets up the neurons in the network with initial values for weights and connections.
- #### `initializeWeights(float* weights, int max_neurons, int max_connections, float* input_tensor)` Initializes the weights for the neural network. Sets the initial weights for the connections between neurons.
- #### `updateNeuronsWithPredictiveCoding(Neuron* neurons, float* input_tensor, int max_neurons, float learning_rate)` Updates neurons using predictive coding. Adjusts the neurons based on the difference between predicted and actual inputs.
- #### `updateWeights(float* weights, Neuron* neurons, int* connections, float learning_rate)` Updates the weights based on learning rate. Adjusts the weights of the connections between neurons to improve the network's performance.
- #### `updateBidirectionalWeights(float* weights, float* reverse_weights, Neuron* neurons, int* connections, int* reverse_connections, float learning_rate)` Updates bidirectional weights for reverse processing. Adjusts weights for connections that go in both directions between neurons.
- #### `computePredictionErrors(Neuron* neurons, float* input_tensor, int max_neurons)` Computes prediction errors for the neurons. Calculates the difference between the predicted and actual outputs of the neurons.
- #### `generatePredictiveInputs(float* predictive_inputs, NetworkStateSnapshot* previous_state, int max_neurons)` Generates predictive inputs based on previous states. Creates inputs for the network based on its previous state.
- #### `selectOptimalDecisionPath(Neuron* neurons, float* weights, int* connections, float* input_tensor, int max_neurons, float* previous_outputs, NetworkStateSnapshot* stateHistory, int step, MemoryEntry* relevantMemory, DynamicParameters* params)` Selects the optimal decision path based on current states and parameters. Chooses the best course of action based on the network's current state and relevant memories.
- #### `computeRegionPerformanceMetrics(NetworkPerformanceMetrics* performanceMetrics, Neuron* neurons, float* target_outputs, int max_neurons)` Computes performance metrics for different regions of the network. Evaluates how well different parts of the network are performing.
- #### `updateMetaControllerPriorities(MetaController* metaController, NetworkPerformanceMetrics* performanceMetrics, MetacognitionMetrics* metacognition)` Updates meta-controller priorities based on performance metrics. Adjusts the priorities of the meta-controller based on the network's performance.
- #### `applyMetaControllerAdaptations(Neuron* neurons, float* weights, MetaController* metaController, int max_neurons)` Applies adaptations from the meta-controller to the network. Makes changes to the network based on the meta-controller's instructions.
- #### `selectOptimalMetaDecisionPath(Neuron* neurons, float* weights, int* connections, float* input_tensor, int max_neurons, MetaLearningState* meta_learning_state, MetacognitionMetrics* metacognition)` Selects the optimal meta-decision path based on meta-learning state and metacognition metrics. Chooses the best course of action based on the meta-learning state and metacognition metrics.
- #### `adaptNetworkDynamic(Neuron* neurons, float* weights, DynamicParameters* params, float performance_delta, float* input_tensor)` Adapts the network dynamically based on performance delta and input tensor. Makes real-time adjustments to the network based on its performance.
- #### `void freeSpecializationSystem(NeuronSpecializationSystem *system)` Frees memory allocated for the specialization system. Releases resources used by the specialization system.
- #### `void printSpecializationStats(NeuronSpecializationSystem *system)` Prints statistics about the specialization system. Displays information about the specialization of the network's neurons.
- #### `float evaluateSpecializationEffectiveness(NeuronSpecializationSystem, *system float network_performance)` Evaluates the effectiveness of the specialization system. Calculates the overall effectiveness of the specialization system based on network performance.
- #### `void updateSpecializationImportance(NeuronSpecializationSystem *system, float network_performance, float error_rate, Neuron *neurons)` Updates the importance of the specialization system. Adjusts the importance of the specialization system based on network performance and error rate.
- #### `void applySpecializations(NeuronSpecializationSystem *system, Neuron *neurons, float *weights, int *connections, int max_neurons, int max_connections)` Applies specializations to the network. Makes changes to the network based on the specialization system's instructions.
- #### `void detectSpecializations(NeuronSpecializationSystem *system, Neuron *neurons, int max_neurons, float *input_tensor, float *target_outputs, float *previous_outputs, float *previous_states)` Detects specializations in the network. Identifies regions of the network that need special attention.

## Dynamic Parameters and Optimization

- #### `initDynamicParameters()` Initializes dynamic parameters for the system. Sets up parameters that can change over time.
- #### `updateDynamicParameters(DynamicParameters* params, float performance_delta, float stability, float error_rate)` Updates dynamic parameters based on performance delta, stability, and error rate. Adjusts the parameters based on the network's performance and stability.
- #### `optimizeParameters(OptimizationState* opt_state, PerformanceMetrics* performance_history, int step)` Optimizes parameters based on performance history. Finds the best values for the parameters based on past performance.
- #### `analyzeNetworkPerformance(PerformanceMetrics* performance_history, int step)` Analyzes network performance and generates insights. Evaluates the network's performance over time and provides insights.
- #### `generatePerformanceGraph(PerformanceMetrics* performance_history, int step)` Generates a performance graph based on performance history. Creates a visual representation of the network's performance over time.

## Context and Reflection

- #### `updateGlobalContext(GlobalContextManager* contextManager, Neuron* neurons, int max_neurons, float* input_tensor)` Updates the global context based on current network state. Adjusts the global context to reflect the current state of the network.
- #### `integrateGlobalContext(GlobalContextManager* contextManager, Neuron* neurons, int max_neurons, float* weights, int max_connections)` Integrates global context into network processing. Uses the global context to influence the network's processing.
- #### `integrateReflectionSystem(Neuron* neurons, MemorySystem* memorySystem, NetworkStateSnapshot* stateHistory, int step, float* weights, int* connections, ReflectionParameters* reflection_params)` Integrates the reflection system into the network processing. Uses the reflection system to influence the network's processing.
- #### `updateIdentity(SelfIdentitySystem* identity_system, Neuron* neurons, int max_neurons, MemorySystem* memorySystem, float* input_tensor)` Updates the identity system based on current states. Adjusts the identity system to reflect the current state of the network and memories.
- #### `verifyIdentity(SelfIdentitySystem* identity_system)` Verifies the consistency of the identity system. Checks the identity system for errors or inconsistencies.
- #### `analyzeIdentitySystem(SelfIdentitySystem* identity_system)` Analyzes the identity system for potential issues. Evaluates the identity system for problems or areas of improvement.
- #### `createIdentityBackup(SelfIdentitySystem* identity_system)` Creates a backup of the identity system. Saves the current state of the identity system for later restoration.
- #### `restoreIdentityFromBackup(SelfIdentitySystem* identity_system, SelfIdentityBackup* backup)` Restores the identity system from a backup. Loads a previously saved state of the identity system.
- #### `freeIdentityBackup(SelfIdentityBackup* backup)` Frees the memory allocated for the identity backup. Deallocates memory to prevent memory leaks.
- #### `generateIdentityReflection(SelfIdentitySystem* identity_system)` Generates a reflection based on the identity system. Creates a summary or evaluation of the identity system.

## Motivation and Goals

- #### `updateMotivationSystem(IntrinsicMotivation* motivation, float performance_delta, float novelty, float task_difficulty)` Updates the motivation system based on performance delta, novelty, and task difficulty. Adjusts the motivation system based on the network's performance and the difficulty of the task.
- #### `addGoal(GoalSystem* goalSystem, const char* description, float priority)` Adds a new goal to the goal system. Creates a new goal with a given description and priority.
- #### `evaluateGoalProgress(Goal* goal, Neuron* neurons, float* target_outputs)` Evaluates the progress of a goal. Checks how close the network is to achieving the goal.

## Security and Validation

- #### `validateCriticalSecurity(Neuron* neurons, float* weights, int* connections, int max_neurons, int max_connections, MemorySystem* memorySystem)` Validates and prevents things that the network shouldn't do.

## Knowledge and Insights

- #### `integrateKnowledgeFilter(KnowledgeFilter* knowledge_filter, MemorySystem* memorySystem, Neuron* neurons, float* input_tensor)` Integrates a knowledge filter into the system. Uses the knowledge filter to influence the network's processing.
- #### `updateKnowledgeSystem(Neuron* neurons, float* input_tensor, MemorySystem* memory_system, KnowledgeFilter* filter, float current_performance)` Updates the knowledge system based on current performance. Adjusts the knowledge system to reflect the network's current performance.
- #### `printCategoryInsights(KnowledgeFilter* knowledge_filter)` Prints insights from the knowledge filter. Provides information or insights generated by the knowledge filter.

## Internal Self-Expression System

- #### `addSymbol(int symbol_id, const char* description)` Adds a symbol to the internal self-expression system. Creates a new symbol with a given ID and description.
- #### `addQuestion(int question_id, int symbol_ids[], int num_symbols)` Adds a question to the internal self-expression system. Creates a new question using the provided symbols.
- #### `askQuestion(int question_id, Neuron* neurons, float* input_tensor, MemorySystem* memorySystem, float* learning_rate)` Asks a question to the internal self-expression system. Processes the question and generates a response.
- #### `expandMemoryCapacity(MemorySystem *memorySystem)` Expands the memory capacity of the memory system. Increases the amount of memory available in the system.
- #### `adjustBehaviorBasedOnAnswers(Neuron* neurons, float* input_tensor, MemorySystem* memorySystem, float *learning_rate, float *input_noise_scale, float *weight_noise_scale)` Adjusts the behavior based on answers from the internal self-expression system. Makes changes to the network based on the responses to questions.

## Moral Compass

- #### `recordDecisionOutcome(MoralCompass *compass, int principle_index, bool was_ethical)` Records the outcome of a decision based on ethical principles. Updates the moral compass based on the decision outcome.
- #### `resolveEthicalDilemma(MoralCompass *compass, float *decision_options, int num_options, int vector_size)` Resolves ethical dilemmas based on the moral compass. Makes a decision based on ethical principles.
- #### `applyEthicalConstraints(MoralCompass *compass, Neuron *neurons, int max_neurons, float *weights, int max_connections)` Applies ethical constraints to the network. Adjusts the network's weights to align with ethical principles.
- #### `generateEthicalReflection(MoralCompass *compass)` Generates an ethical reflection report based on the current state of the moral compass.
- #### `adaptEthicalFramework(MoralCompass *compass, float learning_rate)` Adapts the ethical framework based on the current state of the network. Adjusts the importance of principles based on their violations and activations.
- #### `freeMoralCompass(MoralCompass *compass)` Frees the memory allocated for the moral compass. Releases the memory used by the moral compass.

## Affective System

- #### `float computeEmotionDistance(EmotionVector *a, EmotionVector *b)` Computes the distance between two emotion vectors. Calculates the difference across valence, arousal, dominance, complexity, and temporal depth dimensions.
- #### `void updateEmotionMomentum(EmotionVector *current, EmotionVector *target, float dt)` Updates emotion momentum based on target emotion. Adjusts current emotion state using momentum forces toward the target state.
- #### `uint32_t findNearestAttractor(AffectiveSystem *sys, float *context)` Finds the nearest emotion attractor to the current state. Searches attractors and considers context alignment and basin strength to determine the closest match.
- #### `void updateAttractorDynamics(AffectiveSystem *sys, float *context, uint32_t step)` Updates attractor dynamics based on context and current state. Manages transitions between attractors, updates visit counts, and adjusts emotional state toward attractor center.
- #### `AttachmentBond *findOrCreateBond(AffectiveSystem *sys, uint32_t entity_id, const char *name)` Finds or creates an attachment bond with a specific entity. Returns existing bond or initializes a new one with default attachment parameters.
- #### `void updateAttachmentBond(AffectiveSystem *sys, AttachmentBond *bond, float interaction_valence, float behavior_alignment, float emotional_exchange)` Updates an attachment bond based on interaction outcomes. Adjusts trust, familiarity, emotional resonance, conflict history, and attachment strength.
- #### `void reshapeEmbeddingsWithEmotion(AffectiveSystem *sys, float *embeddings, uint32_t embed_dim)` Reshapes embeddings based on current emotional state. Modifies input embeddings using valence, arousal, dominance, and attractor influences.
- #### `void integrateAttachmentsIntoIdentity(AffectiveSystem *aff, float *identity_core_values, uint32_t num_values)` Integrates attachment bonds into the identity system. Updates core values based on care investment, trust, and loss cost from bonds.
- #### `void updatePredictiveCommitment(AffectiveSystem *aff, SocialSystem *social_sys, float prediction_error)` Updates predictive commitment based on prediction error. Adjusts commitment weight and triggers conflict emotions when predictions fail.
- #### `void updateAffectiveComplexity(AffectiveSystem *sys, uint32_t step)` Updates affective complexity based on recent history. Analyzes emotional diversity, conflict history, and subconscious influence to adjust complexity.
- #### `void reinforceAttractorFromBond(AffectiveSystem *sys, AttachmentBond *bond, bool positive_interaction)` Reinforces an attractor based on bond interaction. Strengthens basin and stability for current attractor during significant bond interactions.
- #### `float h_iga(SocialSystem *social_sys, AffectiveSystem *aff_sys, EmotionalSystem *emo_sys, int person_id)` Models the gap between performed emotion and actual affective state (social mask layer). Returns mask intensity based on social awareness, empathy, and relationship pressure.
- #### `void simulateEmotionalTrajectory(AffectiveSystem *sys, SocialSystem *social_sys, float *context, int steps)` Simulates emotional trajectory over time. Runs emotional dynamics, attractor transitions, and social masking over specified steps.
- #### `void printAttractorAnalysis(AffectiveSystem *sys)` Prints analysis of emotion attractor basins. Displays basin strength, visit counts, duration, pathology status, and linked attractors.
- #### `void freeAffectiveSystem(AffectiveSystem *a)` Frees memory allocated for the affective system. Releases memory used by attractors, affective embeddings, and attachment bonds.

## Emotion System

- #### `void freeEmotionalSystem(EmotionalSystem *system)` Frees the memory allocated for the emotional system. Releases the memory used by the emotional system.
- #### `void printEmotionalState(EmotionalSystem *system)` Prints the current emotional state of the network. Displays the emotional state of the network, including love, hate, cognitive impact, and emotional regulation.
- #### `void detectEmotionalTriggers(EmotionalSystem *system, Neuron *neurons, float *target_outputs, int num_neurons, unsigned int timestamp)` Detects emotional triggers in the network. Identifies and responds to emotional triggers, such as love, hate, cognitive impact, and emotional regulation.
- #### `void applyEmotionalProcessing(EmotionalSystem *system, Neuron *neurons, int num_neurons, float *input_tensor, float learning_rate, float plasticity)` Applies emotional processing to the network. Adjusts the network's weights and biases based on emotional triggers and emotional state.
- #### `float calculateEmotionalBias(EmotionalSystem *system, float *input, int input_size)` Calculates the emotional bias based on the emotional state and input. Computes the emotional bias for the given input and emotional state.
- #### `void updateEmotionalMemory(EmotionalSystem *system)` Updates the emotional memory of the network. Adds new emotional memories to the memory system and updates the emotional memory index.
- #### `void triggerEmotion(EmotionalSystem *system, int emotion_type, float trigger_strength, unsigned int timestamp)` Triggers an emotional response in the network. Activates a specific emotional response in the network.

## Imagination System

- #### `ImaginationScenario createScenario(Neuron *neurons, MemorySystem *memory_system, int max_neurons, float divergence)` Creates a new imagination scenario. Generates a new imagination scenario based on the current state of the network.
- #### `void simulateScenario(ImaginationScenario *scenario, Neuron *neurons, float *input_tensor, int max_neurons, int steps)` Simulates an imagination scenario. Runs the imagination scenario for the specified number of steps.
- #### `void evaluateScenarioPlausibility(ImaginationScenario *scenario, MemorySystem *memory_system)` Evaluates the plausibility of an imagination scenario. Checks the plausibility of the scenario based on the memory system.
- #### `float applyImaginationToDecision(ImaginationSystem *imagination, Neuron *neurons, float *input_tensor, int max_neurons)` Applies imagination to a decision. Modifies the decision based on the imagination system.
- #### `void updateImaginationCreativity(ImaginationSystem *imagination, float performance_delta, float novelty)` Updates the creativity of the imagination system. Adjusts the creativity of the imagination system based on performance and novelty.
- #### `void freeImaginationSystem(ImaginationSystem *system)` Frees the memory allocated for the imagination system. Releases the memory used by the imagination system.
- #### `void blendImaginedOutcomes(ImaginedOutcome *outcomes, int num_outcomes, float *result_vector)` Blends the imagined outcomes into a result vector. Combines the imagined outcomes to create a result vector.
- #### `bool isScenarioCoherent(ImaginationScenario *scenario, float threshold)` Checks if an imagination scenario is coherent. Determines if the scenario is coherent based on a specified threshold.
- #### `void adjustNeuronsWithImagination(Neuron *neurons, ImaginedOutcome *outcome, int max_neurons, float influence)` Adjusts the activations of neurons based on imagined outcomes. Modifies the activations of the neurons based on the imagined outcomes.

## Social System

- #### `void freeSocialSystem(SocialSystem *system)` Frees the memory allocated for the social system. Deallocates memory to prevent memory leaks.
- #### `char *generateSocialFeedback(SocialSystem *system, const char *context)` Generates social feedback based on the given context. Provides feedback or responses based on social interactions and context.
- #### `void applySocialInfluence(SocialSystem *system, Neuron *neurons, float *weights, int max_neurons)` Applies social influence to the network. Adjusts the neurons and weights based on social interactions and influences.
- #### `void predictBehavior(SocialSystem *system, int person_id, const char *context, float *predicted_behavior)` Predicts the behavior of a person in a given context. Estimates how a person will behave based on social data and context.
- #### `void recordSocialInteraction(SocialSystem *system, int person_id, float *emotional_state, float cooperation_level, float satisfaction, const char *type, const char *context)` Records a social interaction for a person. Stores details about the interaction, including emotional state, cooperation level, satisfaction, type, and context.
- #### `float calculateInteractionDiversity(SocialSystem *system)` Calculates the diversity of social interactions. Measures how varied the social interactions are within the system.
- #### `float negotiateOutcome(SocialSystem* system, int person_id, float* goals, float* other_goals, float* compromise)` Negotiates an outcome between a person and others. Determines a compromise based on the goals of the person and others involved.
- #### `void updatePersonModel(SocialSystem* system, int person_id, float* observed_behavior, float* predicted_behavior)` Updates the model of a person based on observed and predicted behavior. Adjusts the person's model to better reflect their actual behavior.
- #### `void updateEmpathy(SocialSystem* system, EmotionalSystem* emotional_system)` Updates the empathy levels in the social system based on the emotional system. Adjusts empathy to reflect the emotional state of the system.

## Utility Functions

- #### `getCurrentTime()` Returns the current time. Provides the current date and time.
- #### `computeMSELoss(Neuron* neurons, float* target_outputs, int max_neurons)` Computes the mean squared error loss. Calculates the difference between the network's outputs and the target outputs.
- #### `verifyNetworkState(Neuron* neurons, TaskPrompt* current_prompt)` Verifies the current state of the network. Checks the network's state for errors or inconsistencies.
- #### `transformOutputsToText(float* previous_outputs, int max_neurons, char* outputText, size_t size)` Transforms numerical outputs to text. Converts the network's outputs into a readable format.
- #### `findSimilarMemoriesInCluster(MemorySystem* memorySystem, float* vector, float similarity_threshold, int* num_matches)` Finds similar memories in a cluster. Searches for memories that are similar to a given vector.
- #### `captureNetworkState(Neuron* neurons, float* input_tensor, NetworkStateSnapshot* stateHistory, float* weights, int step)` Captures the current state of the network. Saves the network's state for later use.
- #### `printNetworkStates(Neuron* neurons, float* input_tensor, int step)` Prints the current state of the network. Provides information about the network's current state.
- #### `saveNetworkStates(NetworkStateSnapshot* stateHistory, int steps)` Saves the network states to a file. Writes the network's state history to a file for later use.
- #### `printReplayStatistics(MemorySystem* memorySystem)` Prints statistics related to memory replay. Provides information about how often memories are replayed.
- #### `addEmbedding(const char* text, float* embedding)` Adds an embedding for a given text. Converts text into a numerical format that the network can use.
- #### `initializeEmbeddings()` Initializes embeddings for text inputs. Sets up the embeddings for use with the network.
- #### `updateEmbeddings(float* embeddings, float* input_tensor, int max_embeddings, int max_neurons)` Updates embeddings based on input tensor. Adjusts the embeddings to better represent the input data.
- #### `isWordMeaningful(const char* word)` Checks if a word is meaningful. Determines whether a word is relevant or important.
- #### `importPretrainedEmbeddings(const char* embedding_file)` Import pretrained embeddings. Loads pre-trained embeddings from a file for use with the network.

## Initialization Functions

- #### `initializeMetaController(int network_regions)` Initializes the meta-controller. Sets up the meta-controller with initial values for each network region.
- #### `initializeMotivationSystem()` Initializes the motivation system. Sets up the motivation system with default values.
- #### `initializeGoalSystem(int num_goals)` Initializes the goal system. Sets up the goal system with a specified number of goals.
- #### `initializeGlobalContextManager(int max_neurons)` Initializes the global context manager. Sets up the global context manager with initial values for each neuron.
- #### `initializePerformanceMetrics(int network_regions)` Initializes performance metrics. Sets up performance metrics for each network region.
- #### `initializeReflectionParameters()` Initializes reflection parameters. Sets up the reflection parameters with default values.
- #### `initializeSelfIdentity(int num_values, int num_beliefs, int num_markers, int history_size, int pattern_size)` Initializes the self-identity system. Sets up the self-identity system with initial values for values, beliefs, markers, history, and patterns.
- #### `initializeKnowledgeFilter(int size)` Initializes the knowledge filter. Sets up the knowledge filter with a specified size.
- #### `initializeMetacognitionMetrics()` Initializes metacognition metrics. Sets up the metacognition metrics with default values.
- #### `initializeMetaLearningState(int size)` Initializes the meta-learning state. Sets up the meta-learning state with a specified size.
- #### `createWorkingMemorySystem(int capacity)` Creates a working memory system. Sets up a working memory system with a specified capacity.
- #### `initializeMoralCompass(int num_principles)` Initializes the moral compass. Sets up the moral compass with a specified number of principles.
- #### `AffectiveSystem *initializeAffectiveSystem(uint32_t embed_dim)` Initializes a new affective system with a specified embedding dimension. Sets up emotion vectors, attractors, attachment bonds, and affective embeddings with default values.
- #### `EmotionalSystem *initializeEmotionalSystem()` Initializes the emotional system. Sets up the emotional system with default values.
- #### `ImaginationSystem* initializeImaginationSystem(float creativity_factor, float coherence_threshold)` Initializes the imagination system. Creates a new imagination system with the specified creativity factor and coherence threshold.
- #### `SocialSystem* initializeSocialSystem(int max_interactions, int max_models)` Initializes a new social system with specified maximum interactions and models. Sets up the social system with initial values for interactions and models.
- #### `NeuronSpecializationSystem *initializeSpecializationSystem(float threshold)` Initializes the specialization system. Sets up the specialization system with a specified threshold.

## Validation System

- #### `isValidMemoryRegion(void *ptr, size_t size)` Validates a memory block to ensure it is accessible and correctly allocated.
- #### `validateMemoryBlock(void *ptr, size_t expected_size, const char *component_name)` Validates a memory block with additional checks and detailed error reporting.
- #### `segfault_handler(int sig, siginfo_t *si, void *unused)` Handles segmentation faults by capturing fault details and attempting recovery.
- #### `initializeSegfaultProtection()` Initializes protection against segmentation faults by setting up signal handlers.
- #### `validateWorkingMemory(WorkingMemorySystem *wm)` Validates the working memory system to ensure it is correctly allocated and within expected bounds.
- #### `validateMetaController(MetaController *mc)` Validates the meta-controller to ensure its parameters and memory regions are correct.
- #### `validatePerformanceMetrics(NetworkPerformanceMetrics *npm)` Validates the network performance metrics to ensure they are correctly allocated and contain valid data.
- #### `validateMotivationSystem(IntrinsicMotivation *im)` Validates the intrinsic motivation system to ensure its parameters are within valid ranges.
- #### `validateReflectionParameters(ReflectionParameters *rp)` Validates the reflection parameters to ensure they are correctly set and within valid ranges.
- #### `validateIdentitySystem(SelfIdentitySystem *sis)` Validates the self-identity system to ensure its components and memory regions are correct.
- #### `validateKnowledgeFilter(KnowledgeFilter *kf)` Validates the knowledge filter to ensure its components and memory regions are correctly allocated.
- #### `validateMetacognition(MetacognitionMetrics *mm)` Validates the metacognition metrics to ensure they are correctly allocated and contain valid data.
- #### `validateMetaLearning(MetaLearningState *mls)` Validates the meta-learning state to ensure its components and memory regions are correct.
- #### `validateSocialSystem(SocialSystem *ss)` Validates the social system to ensure its components and memory regions are correctly allocated.
- #### `validateGoalSystem(GoalSystem *gs)` Validates the goal system to ensure its components and memory regions are correctly allocated.
- #### `validateContextManager(GlobalContextManager *gcm)` Validates the global context manager to ensure its components and memory regions are correctly allocated.
- #### `validateEmotionalSystem(EmotionalSystem *es)` Validates the emotional system to ensure its components and memory regions are correctly allocated.
- #### `validateImaginationSystem(ImaginationSystem *is)` Validates the imagination system to ensure its components and memory regions are correctly allocated.
- #### `validateSpecializationSystem(NeuronSpecializationSystem *nss)` Validates the neuron specialization system to ensure its components and memory regions are correctly allocated.
- #### `validateMoralCompass(MoralCompass *mc)` Validates the moral compass to ensure its components and memory regions are correctly allocated.
- #### `checkMemoryCluster(MemoryCluster *cluster, const char *name)` Checks a memory cluster for validity and consistency, with recovery mechanisms for corrupted entries.
- #### `checkSystemComponent(void *component, const char *name, size_t expected_size)` Comprehensive system component checker that validates various system components based on their expected size and type.
- #### `checkMemoryUsage()` Checks the current memory usage and reports any issues or high usage.
- #### `logSystemState()` Logs the current system state for debugging purposes, including memory usage and timestamp.
- #### `saveEmergencyBackup()` Saves an emergency backup of the system state to a file for recovery purposes.
- #### `stabilizeSystem()` Attempts to stabilize the system after a critical failure or high instability.
- #### `attemptSystemRecovery(const char *failure_description)` Attempts to recover the system from a critical failure by logging the state, saving a backup, and stabilizing the system.
- #### `validateMemoryRegionDetailed(void *ptr, size_t size, const char *region_name)` Enhanced memory region validator with detailed analysis and reporting.
- #### `fpe_handler(int sig, siginfo_t *si, void *unused)` Handles floating-point exceptions by capturing exception details and attempting recovery.
- #### `setupEnhancedSignalHandlers()` Sets up enhanced signal handlers for segmentation faults and floating-point exceptions with detailed error reporting.
- #### `initializeSystemHealthMonitor()` Initializes the system health monitor to track various health metrics and set up signal handlers.
- #### `updateHealthMetrics(bool check_passed, double check_duration)` Updates the health metrics based on the outcome and duration of system checks.
- #### `printSystemHealthReport()` Prints a comprehensive report of the system's health, including uptime, check statistics, and critical events.
- #### `systemFallbackCheck(...)` Performs a comprehensive fallback check of the system, validating various components and handling any critical errors or inconsistencies.

## Notes 

The metal version is more experimental in the sense of the structure of the int main function you should mostly use the C version contained in 64x/main folder by compiling it as a library and then running it through ctypes.

You can call ./neural_web {dataset} to load a dataset.

Note if you modify max_neurons in the example you have to also modify the input_size to be at max greater than the number of max_neurons by 1 or just lesser than the number of max_neurons or it will have an out of bounds error

The model uses reverse pathways and generally doesn't only do patterns good because it also reverses its outputs and finds more meaning in it and additional pathways to achieve what it is supposed to similar to how humans do, or as how I think humans do, is very dynamic and has meta cognition capabilities.

See Documents folder for more information on the neural web.

To modify number of neurons change MAX_NEURONS

You can use vocabulary the converter assuming this json structure {"WORD": {"MEANINGS": [[...]], "ANTONYMS": [...], "SYNONYMS": [...]}} it will then output the correct structure for the neural web to read.

Remember to use the security feature

Only unix systems.

