#include "node_nn/nn.h"
#include "node_nn/utils/io.h"
#include <iostream>
#include <vector>

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

    node_nn::AdamState state;

    std::vector<std::array<float, 7>> inputs;
    inputs.push_back(input);
    std::vector<std::array<float, 5>> outputs;
    outputs.push_back(target);


    // Online update (single step)
    node_nn::adam(nn, inputs, outputs, state);

    // Forward pass after one update
    node_nn::forward(nn, input, output);
    std::cout << "Output after one update: ";
    for (float v : output) std::cout << v << ' ';
    std::cout << std::endl;

    // Optionally, run more updates to see learning and print error for each step
    float error = 0.0f;
    for (int i = 0; i < 1000; ++i) {
        node_nn::adam(nn, inputs, outputs, state);
        // Calculate error after update
        node_nn::forward(nn, input, output);
        node_nn::cost(output, target, error);
        std::cout << "Error after step " << i+1 << ": " << error << std::endl;
    }
    node_nn::forward(nn, input, output);
    std::cout << "Output after 1000 updates: ";
    for (float v : output) std::cout << v << ' ';
    std::cout << std::endl;

    std::string model_path = "test_model.nn";
    if (node_nn::save_model(model_path, nn)) {
        std::cout << "Model saved to " << model_path << std::endl;
    }

    node_nn::NeuralNetwork loaded_nn;

    if (node_nn::load_model(model_path, loaded_nn)) {
        std::cout << "Model loaded from " << model_path << std::endl;
    }

    // 5. 読み込んだモデルで推論して結果を比較
    std::array<float, node_nn::OUTPUT_SIZE> loaded_output = {};
    node_nn::forward(loaded_nn, input, loaded_output);

    std::cout << "Loaded output:   ";
    for (float v : loaded_output) std::cout << v << ' ';
    std::cout << std::endl;

    // 一致確認
    bool match = true;
    for(int i=0; i<node_nn::OUTPUT_SIZE; ++i) {
        if (std::abs(output[i] - loaded_output[i]) > 1e-6) match = false;
    }
    std::cout << (match ? "SUCCESS: Outputs match!" : "FAILURE: Outputs differ!") << std::endl;

    return 0;
}