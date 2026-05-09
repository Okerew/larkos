#include <metal_stdlib>
using namespace metal;

// Constants for neural network parameters
constant float DECAY_RATE = 0.8f;
constant float INPUT_WEIGHT = 0.1f;
constant float CONNECTION_WEIGHT = 0.2f;
constant float ACTIVATION_SCALE = 1.5f;
constant float ACTIVATION_BIAS = 0.1f;
constant float MIN_ACTIVATION = -1.0f;
constant float MAX_ACTIVATION = 1.0f;
constant float LEARNING_RATE = 0.01f;
constant float WEIGHT_DECAY = 0.1f;
constant float MIN_WEIGHT = -1.0f;
constant float MAX_WEIGHT = 1.0f;
constant uint MAX_NEURONS = 20;      // Adjust as needed
constant uint MAX_CONNECTIONS = 12;    // Adjust as needed
constant uint ACTIVATION_TANH = 0;
constant uint ACTIVATION_RELU = 1;
constant uint ACTIVATION_SIGMOID =  2;
constant uint ACTIVATION_LEAKY_RELU =  3;
constant uint ACTIVATION_SWISH = 4;


struct MemoryEntry {
  float vector[MAX_NEURONS * 2]; // Stores neuron states and outputs
  float importance;              // Importance of the memory
  uint timestamp;                // Timestamp of memory creation
};

struct Neuron {
  float state;
  float output;
  uint num_connections;
  uint layer_id;
};

// Fast approximation of tanh for better performance
static inline float fast_tanh(float x) {
  float x2 = x * x;
  float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
  float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
  return clamp(a / b, MIN_ACTIVATION, MAX_ACTIVATION);
}

// ReLU activation function
static inline float relu(float x) {
  return metal::max(0.0f, x);
}

// Sigmoid activation function
static inline float sigmoid(float x) {
  return 1.0f / (1.0f + metal::exp(-x));
}

// Leaky ReLU activation function
static inline float leaky_relu(float x, float alpha = 0.01f) {
  return x > 0.0f ? x : alpha * x;
}

// Swish activation function (x * sigmoid(x))
static inline float swish(float x) {
  return x * sigmoid(x);
}

// Activation function with configurable response curve and type
static inline float activation_function(float x, float scale, float bias, uint activation_type) {
  // Apply scale and bias
  float scaled = x * scale + bias;

  // Select activation function based on type
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
      base_activation = s * fast_tanh(metal::log(1.0f + metal::exp(s)));
      break;
    }
    case ACTIVATION_TANH:
    default:
    base_activation = fast_tanh(scaled);
    break;
  }

  // Add nonlinearity for more dynamic response
  float sign_val = metal::sign(base_activation);
  float abs_val = metal::abs(base_activation);

  // Apply additional nonlinearity for tanh and sigmoid only
  if (activation_type == ACTIVATION_TANH || activation_type == ACTIVATION_SIGMOID) {
    return sign_val * metal::pow(abs_val, 1.1f);
  } else {
    return base_activation;
  }
}

kernel void update_neurons(device Neuron* neurons [[buffer(0)]],
device const float* weights [[buffer(1)]],
device const uint* connections [[buffer(2)]],
device const uint& max_neurons [[buffer(3)]],
device const uint& max_connections [[buffer(4)]],
device const float* input_tensor [[buffer(5)]],
device const uint& input_size [[buffer(6)]],
device const float* recurrent_weights [[buffer(7)]],
device const uint& activation_type [[buffer(8)]],
uint id [[thread_position_in_grid]]) {
  // Early exit for out of bounds threads
  if (id >= max_neurons) return;

  // Load neuron data into thread-local storage
  float current_state = neurons[id].state;
  float current_output = neurons[id].output;
  uint num_conn = neurons[id].num_connections;
  uint layer = neurons[id].layer_id;

  // Calculate weighted sum of inputs from connected neurons
  float weighted_sum = 0.0f;

  // Process connections
  for (uint i = 0; i < num_conn; i++) {
    uint conn_idx = id * max_connections + i;
    uint target = connections[conn_idx];

    // Add weight scaling based on layer depth
    float depth_scale = 1.0f / (1.0f + layer);
    float connection_strength = weights[conn_idx] * depth_scale;

    // Combine state and output influences
    weighted_sum += neurons[target].state * connection_strength * 0.6f +
    neurons[target].output * connection_strength * 0.4f;
  }

  // Calculate input influence with temporal dynamics
  float input_influence = input_tensor[id % input_size];
  float temporal_factor = 1.0f / (1.0f + id % 4); // Creates wave-like temporal patterns

  // Update state with multiple influences
  float new_state = current_state * DECAY_RATE +
  weighted_sum * CONNECTION_WEIGHT +
  input_influence * INPUT_WEIGHT * temporal_factor;

  // Add recurrent connection influence
  float recurrent_influence = current_output * recurrent_weights[id];
  float rec = recurrent_influence * 0.15f;
  rec = rec / (1.0f + metal::abs(rec));
  new_state += rec;


  // Apply activation function with dynamic scaling
  float dynamic_scale = ACTIVATION_SCALE * (1.0f + 0.1f * metal::sin(input_influence * M_PI_F));

  float h1 = activation_function(new_state, dynamic_scale, ACTIVATION_BIAS, activation_type);

  // micro residual refinement (acts like an extra hidden layer)
  float h2 = activation_function(
  h1 * 1.25f + 0.15f * new_state,
  1.1f,
  0.0f,
  activation_type
  );

  // gated residual blend → prevents stagnation + explosion
  float gate = sigmoid(h1 * 0.7f);
  float new_output = metal::mix(h1, h2, gate);


  // Add slight randomization for variability
  float random_val = metal::fract(metal::sin(dot(float2(float(id), new_state),
  float2(12.9898f, 78.233f))) * 43758.5453f);
  float noise_gate = sigmoid(metal::abs(new_output) * 2.0f);
  new_output += random_val * 0.01f * noise_gate;

  // Ensure outputs stay within valid range
  new_output = metal::clamp(new_output, MIN_ACTIVATION, MAX_ACTIVATION);

  // Write back results
  neurons[id].state = new_state;
  neurons[id].output = new_output;
}

// Weight update kernel
kernel void update_weights(device float* weights [[buffer(0)]],
device const Neuron* neurons [[buffer(1)]],
device const uint* connections [[buffer(2)]],
device const float& learning_rate [[buffer(3)]],
device const uint& max_neurons [[buffer(4)]],
device const uint& max_connections [[buffer(5)]],
uint id [[thread_position_in_grid]]) {
  if (id >= max_neurons * max_connections) return;

  uint neuron_idx = id / max_connections;
  uint conn_idx = id % max_connections;

  if (conn_idx >= neurons[neuron_idx].num_connections) return;

  uint target_idx = connections[id];

  // Modified Hebbian learning with normalization
  float pre_activation = neurons[neuron_idx].state;
  float post_activation = neurons[target_idx].output;
  float current_weight = weights[id];

  // Calculate weight update
  float hebbian_term = pre_activation * post_activation;
  float normalization_term = current_weight * WEIGHT_DECAY;
  float delta_w = learning_rate * (hebbian_term - normalization_term);

  // Update weight with momentum
  float momentum = 0.9f;
  float new_weight = current_weight + (1.0f - momentum) * delta_w;

  // Clip weights
  weights[id] = metal::clamp(new_weight, MIN_WEIGHT, MAX_WEIGHT);
}

kernel void backwardKernel(
const device Neuron *neurons [[buffer(0)]],
device float *weights [[buffer(1)]],
const device uint2 *connections [[buffer(2)]],
const device uint *maxNeurons [[buffer(3)]],
const device uint *maxConnections [[buffer(4)]],
const device float *targetOutputs [[buffer(5)]],
device float *outputErrors [[buffer(6)]],
device float *m [[buffer(7)]],   // First moment buffer for Adam
device float *v [[buffer(8)]],   // Second moment buffer for Adam
const device float *beta1 [[buffer(9)]],
const device float *beta2 [[buffer(10)]],
const device float *epsilon [[buffer(11)]],
const device float *learningRate [[buffer(12)]],
const device uint *timeSteps [[buffer(13)]],
device uint *t [[buffer(14)]],  // Optimization step counter
const device uint &activation_type [[buffer(15)]],
uint gid [[thread_position_in_grid]]) {

  if (gid >= *maxNeurons) return;

  float totalError = 0.0;

  // Iterate over time steps in reverse (BPTT)
  for (int step = *timeSteps - 1; step >= 0; step--) {
    uint neuronIndex = gid * (*timeSteps) + step;
    float predictedOutput = neurons[neuronIndex].output;
    float targetOutput = targetOutputs[neuronIndex];

    // Compute error
    float error = predictedOutput - targetOutput;

    float activationGradient;
    float y = predictedOutput;

    switch (activation_type) {
      case ACTIVATION_RELU:
      activationGradient = y > 0.0f ? 1.0f : 0.0f;
      break;
      case ACTIVATION_LEAKY_RELU:
      activationGradient = y > 0.0f ? 1.0f : 0.01f;
      break;
      case ACTIVATION_SWISH:
      activationGradient = y + sigmoid(y) * (1.0f - y);
      break;
      case ACTIVATION_SIGMOID:
      activationGradient = y * (1.0f - y);
      break;
      case ACTIVATION_TANH:
      default:
      activationGradient = 1.0f - y * y;
      break;
    }

    totalError += error;

    // Store error for analysis
    outputErrors[neuronIndex] = error;

    // Compute gradients and update weights
    for (uint i = 0; i < *maxConnections; i++) {
      uint2 connection = connections[gid * (*maxConnections) + i];
      uint connectedNeuron = connection.x;
      uint weightIndex = connection.y;

      if (connectedNeuron >= *maxNeurons) continue;

      float inputGradient = neurons[connectedNeuron * (*timeSteps) + step].output * totalError * activationGradient;

      // Adam Optimizer updates
      float beta1_t = *beta1;
      float beta2_t = *beta2;
      float epsilon_t = *epsilon;
      uint timeStep = *t;

      // Compute first and second moment estimates
      m[weightIndex] = beta1_t * m[weightIndex] + (1.0 - beta1_t) * inputGradient;
      v[weightIndex] = beta2_t * v[weightIndex] + (1.0 - beta2_t) * (inputGradient * inputGradient);

      // Bias correction
      float mHat = m[weightIndex] / (1.0 - pow(beta1_t, timeStep));
      float vHat = v[weightIndex] / (1.0 - pow(beta2_t, timeStep));

      // Adam weight update
      weights[weightIndex] -= (*learningRate) * mHat / (sqrt(vHat) + epsilon_t);
    }
  }

  // Increment the optimization step counter
  atomic_fetch_add_explicit((device atomic_uint *)t, 1, memory_order_relaxed);
}


kernel void reverse_process(
device Neuron *neurons [[buffer(0)]],
device float *reverse_weights [[buffer(1)]],
device uint *reverse_connections [[buffer(2)]],
constant uint &max_neurons [[buffer(3)]],
constant uint &max_connections [[buffer(4)]],
uint id [[thread_position_in_grid]])
{
  if (id >= max_neurons) return;

  float sum = 0.0f;
  for (uint c = 0; c < max_connections; ++c) {
    uint conn_idx = id * max_connections + c;
    uint source_neuron = reverse_connections[conn_idx];

    // Accumulate contributions from reverse connections
    sum += neurons[source_neuron].output * reverse_weights[conn_idx];
  }

  // Update neuron state based on reverse pathway
  neurons[id].state += 0.1f * sum; // Adjust the scaling factor as needed
}

kernel void memory_replay(
device Neuron *neurons [[buffer(0)]],
device float *weights [[buffer(1)]],
device uint *connections [[buffer(2)]],
device MemoryEntry *memories [[buffer(3)]],
constant uint &memory_capacity [[buffer(4)]],
uint id [[thread_position_in_grid]])
{
  if (id >= memory_capacity) return;

  // Retrieve the memory entry
  MemoryEntry memory = memories[id];

  // Reinforce weights based on memory importance
  for (uint i = 0; i < MAX_NEURONS; ++i) {
    for (uint j = 0; j < MAX_CONNECTIONS; ++j) {
      uint conn_idx = i * MAX_CONNECTIONS + j;
      uint target_neuron = connections[conn_idx];

      // Update weights based on memory importance and neuron states
      float weight_delta = 0.01f * memory.importance *
      neurons[i].output *
      memory.vector[target_neuron];
      weights[conn_idx] += weight_delta;
    }
  }
}
