#include "node_nn/nn.h"
#include "node_nn/utils/io.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

std::string find_existing_path(const std::vector<std::string>& candidates) {
    for (const auto& path : candidates) {
        if (std::filesystem::exists(path)) {
            return path;
        }
    }
    return {};
}

void shuffle_training_data(node_nn::TrainingData& data, unsigned int seed) {
    if (data.input.size() != data.target.size() || data.input.empty()) {
        return;
    }

    std::vector<size_t> indices(data.input.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    std::vector<std::array<float, node_nn::INPUT_SIZE>> shuffled_input;
    std::vector<std::array<float, node_nn::OUTPUT_SIZE>> shuffled_target;
    shuffled_input.reserve(data.input.size());
    shuffled_target.reserve(data.target.size());

    for (size_t idx : indices) {
        shuffled_input.push_back(data.input[idx]);
        shuffled_target.push_back(data.target[idx]);
    }

    data.input = std::move(shuffled_input);
    data.target = std::move(shuffled_target);
}

} // namespace

int main(int argc, char* argv[]) {
    std::string csv_path;
    if (argc > 1) {
        csv_path = argv[1];
    } else {
        csv_path = find_existing_path({
            "heuristics/training_data.csv",
            "../heuristics/training_data.csv"
        });
    }

    if (csv_path.empty()) {
        std::cerr << "Could not find training CSV. Provide path as first argument.\n";
        return 1;
    }

    int epochs = 500;
    if (argc > 2) {
        epochs = std::max(1, std::stoi(argv[2]));
    }

    std::string output_model_path = "node_nn_model.nn";
    if (argc > 3) {
        output_model_path = argv[3];
    }

    float test_ratio = 0.2f;
    if (argc > 4) {
        test_ratio = std::stof(argv[4]);
        if (test_ratio < 0.0f) test_ratio = 0.0f;
        if (test_ratio > 0.9f) test_ratio = 0.9f;
    }

    node_nn::TrainingData data;
    if (!node_nn::load_training_data(csv_path, data)) {
        std::cerr << "Failed to load training data from: " << csv_path << "\n";
        return 1;
    }

    if (data.input.empty()) {
        std::cerr << "Training data is empty: " << csv_path << "\n";
        return 1;
    }

    shuffle_training_data(data, 42u);

    node_nn::TrainingData train = data;
    node_nn::TrainingData test;
    node_nn::separate_train_data(train, test, test_ratio);

    if (train.input.empty()) {
        std::cerr << "Training split is empty. Use a smaller test_ratio.\n";
        return 1;
    }

    node_nn::NeuralNetwork nn;
    node_nn::AdamState adam_state;

    std::cout << "Training CSV: " << csv_path << "\n";
    std::cout << "Samples: " << data.input.size() << " (train=" << train.input.size()
              << ", test=" << test.input.size() << ")\n";
    std::cout << "Epochs: " << epochs << ", test_ratio: " << test_ratio << "\n";

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        node_nn::adam(nn, train, adam_state);

        if (epoch == 1 || epoch % 25 == 0 || epoch == epochs) {
            const float train_loss = node_nn::average_cost(nn, train);
            std::cout << "Epoch " << epoch << " | train_loss=" << train_loss;
            if (!test.input.empty()) {
                const float test_loss = node_nn::average_cost(nn, test);
                std::cout << " | test_loss=" << test_loss;
            }
            std::cout << "\n";
        }
    }

    if (!node_nn::save_model(output_model_path, nn)) {
        std::cerr << "Failed to save model: " << output_model_path << "\n";
        return 1;
    }

    std::cout << "Saved model to: " << output_model_path << "\n";
    return 0;
}