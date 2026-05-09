#ifndef EXPOSE_FROM_IMMITRIN
#define EXPOSE_FROM_IMMITRIN
#include "definitions.h"

void updateNeuronStates(Neuron *neurons, int num_neurons,
                        float *recurrent_weights, float scaled_factor);
void initializeWeights(float *weights, int max_neurons, int max_connections,
                       float *input_tensor);
void updateWeights(float *weights, Neuron *neurons, unsigned int *connections,
                   float learning_rate);
void processNeurons(Neuron *neurons, int num_neurons, float *weights,
                    int *connections, int max_connections, float scaled_factor);
#endif // EXPOSE_FROM_IMMITRIN
