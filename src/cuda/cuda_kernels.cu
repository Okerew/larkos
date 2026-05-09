#ifndef NEURAL_KERNELS_CU
#define NEURAL_KERNELS_CU

#include "../../include/definitions.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

typedef struct {
  float state;
  float output;
  unsigned int num_connections;
  unsigned int layer_id;
} Neuron;

typedef struct {
  float vector[MEMORY_VECTOR_SIZE];
  float importance;
  unsigned int timestamp;
} MemoryEntry;

__device__ float dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }

__device__ float fract(float x) { return x - floorf(x); }

__device__ float fast_tanh(float x) {
  float x2 = x * x;
  float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
  float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
  return fminf(fmaxf(a / b, MIN_ACTIVATION), MAX_ACTIVATION);
}

__device__ float relu(float x) { return fmaxf(0.0f, x); }

__device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ float leaky_relu(float x, float alpha = 0.01f) {
  return x > 0.0f ? x : alpha * x;
}

__device__ float swish(float x) { return x * sigmoid(x); }

__device__ float activation_function(float x, float scale, float bias,
                                     unsigned int activation_type) {
  float scaled = x * scale + bias;

  float base_activation;
  switch (activation_type) {
  case ACTIVATION_RELU:
    base_activation = relu(scaled);
    break;
  case ACTIVATION_SIGMOID:
    base_activation = sigmoid(scaled);
    break;
  case ACTIVATION_LEAKY_RELU:
    base_activation = leaky_relu(scaled);
    break;
  case ACTIVATION_SWISH: {
    float s = swish(scaled);
    base_activation = s * fast_tanh(logf(1.0f + expf(s)));
    break;
  }
  case ACTIVATION_TANH:
  default:
    base_activation = fast_tanh(scaled);
    break;
  }

  float sign_val = copysignf(1.0f, base_activation);
  float abs_val = fabsf(base_activation);

  if (activation_type == ACTIVATION_TANH ||
      activation_type == ACTIVATION_SIGMOID) {
    return sign_val * powf(abs_val, 1.1f);
  } else {
    return base_activation;
  }
}

__global__ void
update_neurons(Neuron *neurons, const float *weights,
               const unsigned int *connections, const unsigned int max_neurons,
               const unsigned int max_connections, const float *input_tensor,
               const unsigned int input_size, const float *recurrent_weights,
               const unsigned int activation_type) {
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_neurons)
    return;

  float current_state = neurons[id].state;
  float current_output = neurons[id].output;
  unsigned int num_conn = neurons[id].num_connections;
  unsigned int layer = neurons[id].layer_id;

  float weighted_sum = 0.0f;

  for (unsigned int i = 0; i < num_conn; i++) {
    unsigned int conn_idx = id * max_connections + i;
    unsigned int target = connections[conn_idx];

    float depth_scale = 1.0f / (1.0f + layer);
    float connection_strength = weights[conn_idx] * depth_scale;

    weighted_sum += neurons[target].state * connection_strength * 0.6f +
                    neurons[target].output * connection_strength * 0.4f;
  }

  float input_influence = input_tensor[id % input_size];
  float temporal_factor = 1.0f / (1.0f + id % 4);

  float new_state = current_state * DECAY_RATE +
                    weighted_sum * CONNECTION_WEIGHT +
                    input_influence * INPUT_WEIGHT * temporal_factor;

  float recurrent_influence = current_output * recurrent_weights[id];
  float rec = recurrent_influence * 0.15f;
  rec = rec / (1.0f + fabsf(rec));
  new_state += rec;

  float dynamic_scale =
      ACTIVATION_SCALE * (1.0f + 0.1f * sinf(input_influence * M_PI));

  float h1 = activation_function(new_state, dynamic_scale, ACTIVATION_BIAS,
                                 activation_type);

  float h2 = activation_function(h1 * 1.25f + 0.15f * new_state, 1.1f, 0.0f,
                                 activation_type);

  float gate = sigmoid(h1 * 0.7f);
  float new_output = gate * h2 + (1.0f - gate) * h1;

  float2 hash_input = make_float2(id, new_state);
  float random_val = fract(
      sinf(dot(hash_input, make_float2(12.9898f, 78.233f))) * 43758.5453f);
  float noise_gate = sigmoid(fabsf(new_output) * 2.0f);
  new_output += random_val * 0.01f * noise_gate;

  new_output = fminf(fmaxf(new_output, MIN_ACTIVATION), MAX_ACTIVATION);

  neurons[id].state = new_state;
  neurons[id].output = new_output;
}

__global__ void update_weights(float *weights, const Neuron *neurons,
                               const unsigned int *connections,
                               const float learning_rate,
                               const unsigned int max_neurons,
                               const unsigned int max_connections) {
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_neurons * max_connections)
    return;

  unsigned int neuron_idx = id / max_connections;
  unsigned int conn_idx = id % max_connections;

  if (conn_idx >= neurons[neuron_idx].num_connections)
    return;

  unsigned int target_idx = connections[id];

  float pre_activation = neurons[neuron_idx].state;
  float post_activation = neurons[target_idx].output;
  float current_weight = weights[id];

  float hebbian_term = pre_activation * post_activation;
  float normalization_term = current_weight * WEIGHT_DECAY;
  float delta_w = learning_rate * (hebbian_term - normalization_term);

  float momentum = 0.9f;
  float new_weight = current_weight + (1.0f - momentum) * delta_w;

  weights[id] = fminf(fmaxf(new_weight, MIN_WEIGHT), MAX_WEIGHT);
}

__global__ void backward_kernel(const Neuron *neurons, float *weights,
                                const int *connections,
                                const unsigned int max_neurons,
                                const unsigned int max_connections,
                                const float *target_outputs,
                                float *output_errors, const float learning_rate,
                                const unsigned int activation_type) {
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_neurons)
    return;

  float predicted_output = neurons[id].output;
  float target_output = target_outputs[id];

  float error = predicted_output - target_output;
  output_errors[id] = error;

  float activation_gradient;
  float y = predicted_output;

  switch (activation_type) {
  case ACTIVATION_RELU:
    activation_gradient = y > 0.0f ? 1.0f : 0.0f;
    break;
  case ACTIVATION_LEAKY_RELU:
    activation_gradient = y > 0.0f ? 1.0f : 0.01f;
    break;
  case ACTIVATION_SWISH:
    activation_gradient = y + sigmoid(y) * (1.0f - y);
    break;
  case ACTIVATION_SIGMOID:
    activation_gradient = y * (1.0f - y);
    break;
  case ACTIVATION_TANH:
  default:
    activation_gradient = 1.0f - y * y;
    break;
  }

  for (unsigned int i = 0; i < max_connections; i++) {
    int conn_idx = id * max_connections + i;
    int connected_neuron = connections[conn_idx];

    if (connected_neuron >= max_neurons)
      continue;

    float input_gradient =
        neurons[connected_neuron].output * error * activation_gradient;

    weights[conn_idx] -= learning_rate * input_gradient;
  }
}

__global__ void reverse_process(Neuron *neurons, float *reverse_weights,
                                unsigned int *reverse_connections,
                                const unsigned int max_neurons,
                                const unsigned int max_connections) {
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_neurons)
    return;

  float sum = 0.0f;
  for (unsigned int c = 0; c < max_connections; ++c) {
    unsigned int conn_idx = id * max_connections + c;
    unsigned int source_neuron = reverse_connections[conn_idx];

    sum += neurons[source_neuron].output * reverse_weights[conn_idx];
  }

  neurons[id].state += 0.1f * sum;
}

__global__ void memory_replay(Neuron *neurons, float *weights,
                              unsigned int *connections, MemoryEntry *memories,
                              const unsigned int memory_capacity) {
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= memory_capacity)
    return;

  MemoryEntry memory = memories[id];

  for (unsigned int i = 0; i < MAX_NEURONS; ++i) {
    for (unsigned int j = 0; j < MAX_CONNECTIONS; ++j) {
      unsigned int conn_idx = i * MAX_CONNECTIONS + j;
      unsigned int target_neuron = connections[conn_idx];

      float weight_delta = 0.01f * memory.importance * neurons[i].output *
                           memory.vector[target_neuron];
      weights[conn_idx] += weight_delta;
    }
  }
}

#endif
