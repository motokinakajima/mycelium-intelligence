#include "node_nn/nn.h"
#include "node_nn/utils/io.h"
#include <iostream>
#include <string>

int main(const int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <training_data.csv> <model_output.nn>\n";
        return 1;
    }
    const std::string data_path = argv[1];
    const std::string model_path = argv[2];

    constexpr int NUM_ITER = 1000; // Number of training iterations
    constexpr float TEST_RATIO = 0.2f; // Fraction of data for testing

    node_nn::TrainingData all_data;
    if (!node_nn::load_training_data(data_path, all_data)) {
        std::cerr << "Failed to load training data from " << data_path << std::endl;
        return 1;
    }

    node_nn::TrainingData train_data = all_data, test_data;
    node_nn::separate_train_data(train_data, test_data, TEST_RATIO);

    node_nn::NeuralNetwork nn;
    node_nn::AdamState state;

    // Debug: print initial cost before training
    float initial_train_cost = node_nn::average_cost(nn, train_data);
    float initial_test_cost = node_nn::average_cost(nn, test_data);
    std::cout << "Initial train cost: " << initial_train_cost << ", Initial test cost: " << initial_test_cost << std::endl;

    // Training loop
    for (int i = 0; i < NUM_ITER; ++i) {
        node_nn::adam(nn, train_data, state);
        if ((i+1) % 100 == 0 || i == 0) {
            float train_cost = node_nn::average_cost(nn, train_data);
            float test_cost = node_nn::average_cost(nn, test_data);
            std::cout << "Iteration " << (i+1)
                      << ": Train cost = " << train_cost
                      << ", Test cost = " << test_cost << std::endl;
        }
    }

    if (node_nn::save_model(model_path, nn)) {
        std::cout << "Model saved to " << model_path << std::endl;
    } else {
        std::cerr << "Failed to save model to " << model_path << std::endl;
        return 1;
    }
    return 0;
}