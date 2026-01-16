#pragma once
#include <array>

constexpr int INPUT_SIZE = 7;
constexpr int HIDDEN_SIZE = 6;
constexpr int OUTPUT_SIZE = 5;
constexpr float LEARNING_RATE = 0.1f;

struct NeuralNetwork {
    std::array<std::array<float, INPUT_SIZE>, HIDDEN_SIZE> W1;
    std::array<float, HIDDEN_SIZE> b1;
    std::array<std::array<float, HIDDEN_SIZE>, OUTPUT_SIZE> W2;
    std::array<float, OUTPUT_SIZE> b2;
    NeuralNetwork();
};

float activate(float x);
void forward(NeuralNetwork& nn, const std::array<float, INPUT_SIZE>& x, std::array<float, OUTPUT_SIZE>& y);
void forward(NeuralNetwork& nn, const std::array<float, INPUT_SIZE>& x, std::array<float, OUTPUT_SIZE>& y, std::array<float, HIDDEN_SIZE>& h, const std::array<float, OUTPUT_SIZE>& target, float& error);
void back_propagate(NeuralNetwork& nn, const std::array<float, INPUT_SIZE>& x, const std::array<float, OUTPUT_SIZE>& target);
