#include "include/definitions.h"
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

__m128 _mm_tanh_ps(__m128 x) {
  __m128 result = _mm_setzero_ps();
  float *px = (float *)&x;
  float *pr = (float *)&result;

  for (int i = 0; i < 4; i++) {
    float exp_pos = exp(px[i]);
    float exp_neg = exp(-px[i]);
    pr[i] = (exp_pos - exp_neg) / (exp_pos + exp_neg);
  }

  return result;
}

void updateNeuronStates(Neuron *neurons, int num_neurons,
                        float *recurrent_weights, float scaled_factor) {
  // Process neurons in groups of 4
  for (int i = 0; i < num_neurons; i += 4) {
    // Ensure we don't overrun the array
    int remaining = num_neurons - i;
    int group_size = (remaining < 4) ? remaining : 4;

    // Load outputs, states, and weights for the current group
    __m128 current_outputs = _mm_setzero_ps();
    __m128 current_states = _mm_setzero_ps();
    __m128 current_weights = _mm_setzero_ps();

    // Load data conditionally based on group size
    float outputs[4] = {0}, states[4] = {0}, weights[4] = {0};
    for (int j = 0; j < group_size; j++) {
      outputs[j] = neurons[i + j].output;
      states[j] = neurons[i + j].state;
      weights[j] = recurrent_weights[i + j];
    }
    // Use unaligned loads since arrays may not be 16-byte aligned
    current_outputs = _mm_loadu_ps(outputs);
    current_states = _mm_loadu_ps(states);
    current_weights = _mm_loadu_ps(weights);

    // Update states with decay factor
    __m128 decay_factor = _mm_set1_ps(0.8f);
    __m128 new_states = _mm_mul_ps(current_states, decay_factor);

    // Calculate recurrent inputs
    __m128 recurrent_inputs = _mm_mul_ps(current_outputs, current_weights);

    // Simulate neighbor influence
    __m128 neighbor_influence = current_outputs;

    // Combine influences
    __m128 recurrent_factor = _mm_set1_ps(0.3f);
    __m128 neighbor_factor = _mm_set1_ps(0.2f);
    new_states =
        _mm_add_ps(new_states, _mm_mul_ps(recurrent_inputs, recurrent_factor));
    new_states =
        _mm_add_ps(new_states, _mm_mul_ps(neighbor_influence, neighbor_factor));

    // Apply activation function and scale
    __m128 scale = _mm_set1_ps(scaled_factor);
    __m128 new_outputs = _mm_tanh_ps(_mm_mul_ps(new_states, scale));

    // Store updated values back to neurons
    float result_states[4], result_outputs[4];
    _mm_storeu_ps(result_states, new_states);
    _mm_storeu_ps(result_outputs, new_outputs);

    for (int j = 0; j < group_size; j++) {
      neurons[i + j].state = result_states[j];
      neurons[i + j].output = result_outputs[j];
      if (isnan(neurons[i + j].state))
        neurons[i + j].state = 0.0f;
      if (isnan(neurons[i + j].output))
        neurons[i + j].output = 0.0f;
    }
  }
}

void initializeWeights(float *weights, int max_neurons, int max_connections,
                       float *input_tensor) {
  srand(time(NULL)); // Seed random number generator

  for (int i = 0; i < max_neurons; i++) {
    for (int j = 0; j < max_connections; j++) {
      // Random initialization in range [-0.5, 0.5]
      weights[i * max_connections + j] =
          input_tensor[(i + j) % INPUT_SIZE] * 0.5f - 0.25f;
    }
  }
}

void updateWeights(float *weights, Neuron *neurons, unsigned int *connections,
                   float learning_rate) {
  for (int i = 0; i < MAX_NEURONS; i++) {
    for (int j = 0; j < neurons[i].num_connections; j++) {
      int target_idx = connections[i * MAX_CONNECTIONS + j];
      // Modified Hebbian learning rule with normalization
      float pre_activation = neurons[i].state;
      float post_activation = neurons[target_idx].output;
      float current_weight = weights[i * MAX_CONNECTIONS + j];
      // Calculate weight update with normalization term
      float delta_w = learning_rate *
                      (pre_activation * post_activation - // Hebbian term
                       current_weight * 0.1f // Weight decay for normalization
                      );
      // Update weight
      weights[i * MAX_CONNECTIONS + j] += delta_w;
      // Clip weights to prevent unbounded growth
      weights[i * MAX_CONNECTIONS + j] =
          fmaxf(-1.0f, fminf(1.0f, weights[i * MAX_CONNECTIONS + j]));
    }
  }
}

void processNeurons(Neuron *neurons, int num_neurons, float *weights,
                    int *connections, int max_connections,
                    float scaled_factor) {
  // Process neurons in groups of 4
  for (int i = 0; i < num_neurons; i += 4) {
    // Ensure we don't overrun the array
    int remaining = num_neurons - i;
    int group_size = (remaining < 4) ? remaining : 4;

    __m128 weighted_sum = _mm_setzero_ps();

    // Compute weighted sum for connections
    for (int j = 0; j < max_connections; j++) {
      float weight_array[4] = {0};
      float target_state_array[4] = {0};

      for (int k = 0; k < group_size; k++) {
        int neuron_idx = i + k;
        int connection_idx = connections[neuron_idx * max_connections + j];
        weight_array[k] = weights[neuron_idx * max_connections + j];
        target_state_array[k] = neurons[connection_idx].state;
      }

      // Use unaligned loads since arrays may not be 16-byte aligned
      __m128 weight_vector = _mm_loadu_ps(weight_array);
      __m128 target_state = _mm_loadu_ps(target_state_array);

      weighted_sum =
          _mm_add_ps(weighted_sum, _mm_mul_ps(weight_vector, target_state));
    }

    // Load current states
    float current_state_array[4] = {0};
    for (int k = 0; k < group_size; k++) {
      current_state_array[k] = neurons[i + k].state;
    }
    __m128 current_states = _mm_loadu_ps(current_state_array);

    // Combine with decay factor
    __m128 decay_factor = _mm_set1_ps(0.8f);
    __m128 weighted_factor = _mm_set1_ps(0.2f);
    __m128 new_states = _mm_add_ps(_mm_mul_ps(current_states, decay_factor),
                                   _mm_mul_ps(weighted_sum, weighted_factor));

    // Store and update neuron states
    float result_states[4];
    _mm_storeu_ps(result_states, new_states);

    for (int k = 0; k < group_size; k++) {
      neurons[i + k].state = result_states[k];
      neurons[i + k].output = tanh(result_states[k] * scaled_factor);
      if (isnan(neurons[i + k].state))
        neurons[i + k].state = 0.0f;
      if (isnan(neurons[i + k].output))
        neurons[i + k].output = 0.0f;
    }
  }
}
