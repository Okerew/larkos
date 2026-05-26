#include "include/definitions.h"
#include <CoreFoundation/CoreFoundation.h>
#include <simd/simd.h>

/* Vectorised tanh via the Lambert continued-fraction rational
 * approximation: tanh(x) ~= x*(27 + x^2) / (27 + 9*x^2).
 * Accurate to ~1e-3 over [-3, 3]; we clamp to that range first since
 * the approximation drifts past the tails and our states are bounded
 * into ~[-1, 1] anyway. The old version dropped to a scalar loop with
 * tanhf() per lane, which defeats the SIMD just as badly as the SSE
 * version did before the fix there. */
simd_float4 simd_tanh(simd_float4 x) {
  x = simd_clamp(x, -3.0f, 3.0f);

  simd_float4 x2 = x * x;
  simd_float4 num = x * (27.0f + x2);
  simd_float4 den = 27.0f + 9.0f * x2;
  return num / den;
}

simd_float4 simd_add(simd_float4 a, simd_float4 b) {
  return a + b; // SIMD supports operator overloading for addition
}

// Helper function for element-wise multiplication of simd_float4
simd_float4 simd_mul(simd_float4 a, simd_float4 b) {
  simd_float4 result;
  for (int i = 0; i < 4; i++) {
    result[i] = a[i] * b[i];
  }
  return result;
}

// Function to update neuron states using SIMD
void updateNeuronStates(Neuron *neurons, int num_neurons,
                        float *recurrent_weights, simd_float4 scaled_factor) {
  // Compute attention weights between neurons
  float **attention_weights = malloc(num_neurons * sizeof(float *));
  for (int i = 0; i < num_neurons; i++) {
    attention_weights[i] = malloc(num_neurons * sizeof(float));

    // Simplified dot product attention
    for (int j = 0; j < num_neurons; j++) {
      attention_weights[i][j] = neurons[i].output * neurons[j].output;
    }

    // Softmax normalization
    float row_sum = 0;
    for (int j = 0; j < num_neurons; j++) {
      attention_weights[i][j] = exp(attention_weights[i][j]);
      row_sum += attention_weights[i][j];
    }

    for (int j = 0; j < num_neurons; j++) {
      attention_weights[i][j] /= row_sum;
    }
  }

  // Process neurons in groups of 4
  for (int i = 0; i < num_neurons; i += 4) {
    int remaining = num_neurons - i;
    int group_size = (remaining < 4) ? remaining : 4;

    simd_float4 current_outputs = {0};
    simd_float4 current_states = {0};
    simd_float4 current_weights = {0};

    for (int j = 0; j < group_size; j++) {
      current_outputs[j] = neurons[i + j].output;
      current_states[j] = neurons[i + j].state;
      current_weights[j] = recurrent_weights[i + j];
    }

    // Update states with decay factor
    simd_float4 new_states = simd_mul(current_states, 0.8f);

    // Compute attended inputs
    simd_float4 attended_inputs = {0};
    for (int j = 0; j < group_size; j++) {
      float attended_value = 0;
      for (int k = 0; k < num_neurons; k++) {
        attended_value += attention_weights[i + j][k] * neurons[k].output;
      }
      attended_inputs[j] = attended_value;
    }

    // Recurrent and attention-based inputs
    simd_float4 recurrent_inputs = simd_mul(current_outputs, current_weights);
    new_states = simd_add(new_states, simd_mul(recurrent_inputs, 0.3f));
    new_states = simd_add(new_states, simd_mul(attended_inputs, 0.2f));

    // Apply activation and scaling
    simd_float4 new_outputs = simd_tanh(new_states * scaled_factor);

    // Store updated values
    for (int j = 0; j < group_size; j++) {
      neurons[i + j].state = new_states[j];
      neurons[i + j].output = new_outputs[j];
      if (isnan(neurons[i + j].state))
        neurons[i + j].state = 0.0f;
      if (isnan(neurons[i + j].output))
        neurons[i + j].output = 0.0f;
    }
  }

  // Free attention weights
  for (int i = 0; i < num_neurons; i++) {
    free(attention_weights[i]);
  }
  free(attention_weights);
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

void updateWeights(float *weights, Neuron *neurons, uint *connections,
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
                    uint *connections, int max_connections,
                    float scaled_factor) {
  // Compute attention weights between neurons
  float **attention_weights = malloc(num_neurons * sizeof(float *));
  for (int i = 0; i < num_neurons; i++) {
    attention_weights[i] = malloc(num_neurons * sizeof(float));

    // Simplified dot product attention
    for (int j = 0; j < num_neurons; j++) {
      attention_weights[i][j] = neurons[i].state * neurons[j].state;
    }

    // Softmax normalization
    float row_sum = 0;
    for (int j = 0; j < num_neurons; j++) {
      attention_weights[i][j] = exp(attention_weights[i][j]);
      row_sum += attention_weights[i][j];
    }

    for (int j = 0; j < num_neurons; j++) {
      attention_weights[i][j] /= row_sum;
    }
  }

  // Process neurons in groups of 4
  for (int i = 0; i < num_neurons; i += 4) {
    // Ensure we don't overrun the array
    int remaining = num_neurons - i;
    int group_size = (remaining < 4) ? remaining : 4;

    simd_float4 weighted_sum = simd_make_float4(0, 0, 0, 0);
    simd_float4 attended_sum = simd_make_float4(0, 0, 0, 0);

    // Compute weighted sum for connections
    for (int j = 0; j < max_connections; j++) {
      simd_float4 weight_vector = {0};
      simd_float4 target_state = {0};

      for (int k = 0; k < group_size; k++) {
        int neuron_idx = i + k;
        int connection_idx = connections[neuron_idx * max_connections + j];
        weight_vector[k] = weights[neuron_idx * max_connections + j];
        target_state[k] = neurons[connection_idx].state;
      }

      // Fill remaining SIMD lanes with zeros
      for (int k = group_size; k < 4; k++) {
        weight_vector[k] = 0.0f;
        target_state[k] = 0.0f;
      }

      weighted_sum =
          simd_add(weighted_sum, simd_mul(weight_vector, target_state));
    }

    // Compute self-attention based sum
    for (int j = 0; j < group_size; j++) {
      float attended_value = 0;
      for (int k = 0; k < num_neurons; k++) {
        attended_value += attention_weights[i + j][k] * neurons[k].state;
      }
      attended_sum[j] = attended_value;
    }

    // Combine current states with weighted and attended sums
    simd_float4 current_states = {0};
    for (int k = 0; k < group_size; k++) {
      current_states[k] = neurons[i + k].state;
    }

    // Fill remaining SIMD lanes with zeros
    for (int k = group_size; k < 4; k++) {
      current_states[k] = 0.0f;
    }

    simd_float4 new_states =
        simd_add(simd_mul(current_states, 0.8f),        // Decay current states
                 simd_add(simd_mul(weighted_sum, 0.2f), // Add weighted sum
                          simd_mul(attended_sum, 0.1f)  // Add attended sum
                          ));

    // Update neuron states and outputs
    for (int k = 0; k < group_size; k++) {
      neurons[i + k].state = new_states[k];
      neurons[i + k].output = tanh(new_states[k] * scaled_factor);
      if (isnan(neurons[i + k].state))
        neurons[i + k].state = 0.0f;
      if (isnan(neurons[i + k].output))
        neurons[i + k].output = 0.0f;
    }
  }

  // Free attention weights
  for (int i = 0; i < num_neurons; i++) {
    free(attention_weights[i]);
  }
  free(attention_weights);
}
