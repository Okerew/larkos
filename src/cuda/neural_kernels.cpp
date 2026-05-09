#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <vector>

void update_neurons_wrapper(torch::Tensor neurons, torch::Tensor weights,
                            torch::Tensor connections, int max_neurons,
                            int max_connections, torch::Tensor input_tensor,
                            int input_size, torch::Tensor recurrent_weights,
                            int activation_type);

void update_weights_wrapper(torch::Tensor weights, torch::Tensor neurons,
                            torch::Tensor connections, float learning_rate,
                            int max_neurons, int max_connections);

void backward_wrapper(torch::Tensor neurons, torch::Tensor weights,
                      torch::Tensor connections, int max_neurons,
                      int max_connections, torch::Tensor target_outputs,
                      torch::Tensor output_errors, float learning_rate);

void reverse_process_wrapper(torch::Tensor neurons,
                             torch::Tensor reverse_weights,
                             torch::Tensor reverse_connections, int max_neurons,
                             int max_connections);

void memory_replay_wrapper(torch::Tensor neurons, torch::Tensor weights,
                           torch::Tensor connections, torch::Tensor memories,
                           int memory_capacity);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("update_neurons", &update_neurons_wrapper, "Update neurons");
  m.def("update_weights", &update_weights_wrapper, "Update weights");
  m.def("backward", &backward_wrapper, "Backward pass");
  m.def("reverse_process", &reverse_process_wrapper, "Reverse process");
  m.def("memory_replay", &memory_replay_wrapper, "Memory replay");
}
