#include "nn.h"
#include <iostream>

int main() {
    NeuralNetwork nn;

    // Example input and target
    std::array<float, INPUT_SIZE> input = {0.5f, -0.2f, 0.1f, 0.0f, 0.3f, -0.4f, 0.7f};
    std::array<float, OUTPUT_SIZE> target = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, OUTPUT_SIZE> output;

    // Forward pass before training
    forward(nn, input, output);
    std::cout << "Output before training: ";
    for (float v : output) std::cout << v << ' ';
    std::cout << std::endl;


    // Online update (single step)
    single_back_propagate(nn, input, target);

    // Forward pass after one update
    forward(nn, input, output);
    std::cout << "Output after one update: ";
    for (float v : output) std::cout << v << ' ';
    std::cout << std::endl;

    // Optionally, run more updates to see learning and print error for each step
    float error = 0.0f;
    for (int i = 0; i < 100; ++i) {
        single_back_propagate(nn, input, target);
        // Calculate error after update
        forward(nn, input, output);
        std::cout << "Error after step " << i+1 << ": " << error << std::endl;
    }
    forward(nn, input, output);
    std::cout << "Output after 100 updates: ";
    for (float v : output) std::cout << v << ' ';
    std::cout << std::endl;

    return 0;
}