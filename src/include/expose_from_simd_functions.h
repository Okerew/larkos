#ifndef EXPOSE_FROM_SIMD_FUNCTIONS
#define EXPOSE_FROM_SIMD_FUNCTIONS
#include "definitions.h"
#include <CoreFoundation/CoreFoundation.h>
#include <simd/simd.h>

void updateNeuronStates(Neuron *neurons, int num_neurons,
                        float *recurrent_weights, simd_float4 scaled_factor);
void initializeWeights(float *weights, int max_neurons, int max_connections,
                       float *input_tensor);
void updateWeights(float *weights, Neuron *neurons, uint *connections,
                   float learning_rate);
void processNeurons(Neuron *neurons, int num_neurons, float *weights,
                    uint *connections, int max_connections,
                    float scaled_factor);
#endif // EXPOSE_FROM_SIMD_FUNCTIONS
