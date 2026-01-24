#pragma once
#include <array>
#include <vector>

namespace node_nn {

    constexpr int INPUT_SIZE = 7;
    constexpr int HIDDEN_SIZE = 6;
    constexpr int OUTPUT_SIZE = 5;
    constexpr float LEARNING_RATE = 0.001f;

    constexpr float BETA_1 = 0.9f;
    constexpr float BETA_2 = 0.999f;

    constexpr float EPSILON = 1.0e-8;

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

    struct AdamM : Parameters {
        AdamM();
    };

    struct AdamV : Parameters {
        AdamV();
    };

    struct AdamState {
        AdamM m;
        AdamV v;
        int t;

        AdamState();
    };

    struct TrainingData {
        std::vector<std::array<float, INPUT_SIZE>> input;
        std::vector<std::array<float, OUTPUT_SIZE>> target;
    };

    float activate(float x);

    void forward(const NeuralNetwork &nn, const std::array<float, INPUT_SIZE> &x, std::array<float, OUTPUT_SIZE> &y);

    void forward(const NeuralNetwork &nn, const std::array<float, INPUT_SIZE> &x, std::array<float, OUTPUT_SIZE> &y,
                 std::array<float, HIDDEN_SIZE> &h);

    void cost(const std::array<float, OUTPUT_SIZE> &y, const std::array<float, OUTPUT_SIZE> &target, float &error);

    void add_gradients(const NeuralNetwork &nn, const std::array<float, INPUT_SIZE> &x,
                       const std::array<float, OUTPUT_SIZE> &target, Gradients &gradient);

    void apply_gradients(NeuralNetwork &nn, const Gradients &gradient, float batch_size);

    void single_back_propagate(NeuralNetwork &nn, const std::array<float, INPUT_SIZE> &x,
                               const std::array<float, OUTPUT_SIZE> &target);

    void average_gradients(Gradients &g, float batch_size);

    void back_propagate(NeuralNetwork &nn, const std::array<float, INPUT_SIZE> &input, const std::array<float, OUTPUT_SIZE> &target);

    void back_propagate(NeuralNetwork &nn, const TrainingData &data);

    void adam(NeuralNetwork &nn,
              const std::vector<std::array<float, INPUT_SIZE>> &input,
              const std::vector<std::array<float, OUTPUT_SIZE>> &target,
              AdamState &state);
    
    void adam(NeuralNetwork &nn,
              const TrainingData &data,
              AdamState &state);
    
    void separate_train_data(TrainingData &learn, TrainingData &test, float test_ratio);

}