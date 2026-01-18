#pragma once
#include <array>

namespace node_nn {

    constexpr int INPUT_SIZE = 7;
    constexpr int HIDDEN_SIZE = 6;
    constexpr int OUTPUT_SIZE = 5;
    constexpr float LEARNING_RATE = 0.1f;

    struct Parameters {
        std::array<std::array<float, INPUT_SIZE>, HIDDEN_SIZE> W1;
        std::array<float, HIDDEN_SIZE> b1;
        std::array<std::array<float, HIDDEN_SIZE>, OUTPUT_SIZE> W2;
        std::array<float, OUTPUT_SIZE> b2;

        void set_zero();
        void randomize();
    };

    struct NeuralNetwork : Parameters {
        NeuralNetwork();
    };

    struct Gradients : Parameters {
        Gradients();
    };

    float activate(float x);

    void forward(NeuralNetwork &nn, const std::array<float, INPUT_SIZE> &x, std::array<float, OUTPUT_SIZE> &y);

    void forward(NeuralNetwork &nn, const std::array<float, INPUT_SIZE> &x, std::array<float, OUTPUT_SIZE> &y,
                 std::array<float, HIDDEN_SIZE> &h);

    void cost(const std::array<float, OUTPUT_SIZE> &y, const std::array<float, OUTPUT_SIZE> &target, float &error);

    void add_gradients(const NeuralNetwork &nn, const std::array<float, INPUT_SIZE> &x,
                       const std::array<float, OUTPUT_SIZE> &target, Gradients &gradient);

    void apply_gradients(NeuralNetwork &nn, const Gradients &gradient, float batch_size);

    void single_back_propagate(NeuralNetwork &nn, const std::array<float, INPUT_SIZE> &x,
                               const std::array<float, OUTPUT_SIZE> &target);

    void back_propagate(NeuralNetwork &nn, const std::array<float, INPUT_SIZE> &input, const std::array<float, OUTPUT_SIZE> &target);

}