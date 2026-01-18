#include "nn/nn.h"
#include <iostream>

int main() {
    node_nn::NeuralNetwork nn;

    // Example input and target
    std::array<float, node_nn::INPUT_SIZE> input = {0.5f, -0.2f, 0.1f, 0.0f, 0.3f, -0.4f, 0.7f};
    std::array<float, node_nn::OUTPUT_SIZE> target = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, node_nn::OUTPUT_SIZE> output = {};

    // Forward pass before training
    node_nn::forward(nn, input, output);
    std::cout << "Output before training: ";
    for (float v : output) std::cout << v << ' ';
    std::cout << std::endl;


    // Online update (single step)
    single_back_propagate(nn, input, target);

    // Forward pass after one update
    node_nn::forward(nn, input, output);
    std::cout << "Output after one update: ";
    for (float v : output) std::cout << v << ' ';
    std::cout << std::endl;

    // Optionally, run more updates to see learning and print error for each step
    float error = 0.0f;
    for (int i = 0; i < 1000; ++i) {
        node_nn::single_back_propagate(nn, input, target);
        // Calculate error after update
        node_nn::forward(nn, input, output);
        node_nn::cost(output, target, error);
        std::cout << "Error after step " << i+1 << ": " << error << std::endl;
    }
    node_nn::forward(nn, input, output);
    std::cout << "Output after 1000 updates: ";
    for (float v : output) std::cout << v << ' ';
    std::cout << std::endl;

    return 0;
}