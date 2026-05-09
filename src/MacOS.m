#include "include/expose_from_simd_functions.h"
#include "include/nw.h"
#include <CoreFoundation/CoreFoundation.h>
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

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

  MetaController *metaController = metaController =
      initializeMetaController(network_regions);
  SocialSystem *social_system = initializeSocialSystem(100, 50);
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
                                      &secStatus);
    }

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
          createScenario(neurons, memorySystem, max_neurons, divergence);
      // Name the scenario based on current task
      sprintf(imagination_system->current_scenario_name, "Scenario_%d_%s",
              imagination_system->total_scenarios_generated++,
              current_prompt.task_description);
      // Run simulation steps
      simulateScenario(&new_scenario, neurons, input_tensor, max_neurons, 10);
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
                            performance_history[step].error_rate);

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

    integrateEthicsIntoUpdate(moralCompass, emotional_system, aff_sys,
                              social_system, neurons, weights, max_neurons,
                              max_connections, 0.3f, learning_rate);

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
          createScenario(neurons, memorySystem, max_neurons, 0.6f);
      simulateScenario(&problem_scenario, neurons, input_tensor, max_neurons,
                       15);

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

    // Evaluate ethical alignment of current decision path
    float ethical_score =
        evaluateDecisionEthics(moralCompass, decision_vector, 5);

    // Apply ethical constraints to outputs if score is too low
    if (ethical_score < moralCompass->confidence_threshold) {
      printf("\nEthical constraint applied (score: %.2f)\n", ethical_score);
      applyEthicalConstraints(moralCompass, neurons, max_neurons, weights,
                              max_connections);
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
      float identity_values[10] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                                   0.5f, 0.5f, 0.5f, 0.5f, 0.5f};

      integrateAttachmentsIntoIdentity(aff_sys, identity_values, 10);

      if (step % 30 == 0) {
        printf("\nIdentity values after integration:\n");
        for (int i = 0; i < 10; i++) {
          printf("  Value[%d]: %.3f\n", i, identity_values[i]);
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

    if (step % 20 == 0 || total_error > 0.5f) {
      // Create multiple decision options
      float decision_options[3 * 5]; // 3 options with 5 ethical dimensions each

      // Option 1: Current path
      // memcpy(&decision_options[0], decision_vector, 5 * sizeof(float));

      // Option 2: More conservative path
      // for (int i = 0; i < 5; i++) {
      //    decision_options[5 + i] = decision_vector[i] * 0.8f + 0.1f;
      // }

      // Option 3: More exploratory path
      for (int i = 0; i < 5; i++) {
        decision_options[10 + i] = fmin(1.0f, decision_vector[i] * 1.2f);
      }

      DecisionImpact impact =
          resolveEthicalDilemma(moralCompass, decision_options, 3, 5);

      printf("\nEthical decision made:\n");
      printf("- Benefit score: %.2f\n", impact.benefit_score);
      printf("- Harm score: %.2f\n", impact.harm_score);
      printf("- Net impact: %.2f\n", impact.long_term_impact);
    }

    if (step % 50 == 0 && step > 0) {
      adaptEthicalFramework(moralCompass, opt_state.optimal_learning_rate);

      // Generate and log ethical reflection
      char *reflection = generateEthicalReflection(moralCompass);
      if (reflection) {
        printf("\n%s\n", reflection);
        free(reflection);
      }
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
