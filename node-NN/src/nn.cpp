constexpr int INPUT_SIZE = 7;
constexpr int HIDDEN_SIZE = 6;
constexpr int OUTPUT_SIZE = 5;

static_assert(INPUT_SIZE > 0, "INPUT_SIZE must be greater than 0");
static_assert(HIDDEN_SIZE > 0, "HIDDEN_SIZE must be greater than 0");
static_assert(OUTPUT_SIZE > 0, "OUTPUT_SIZE must be greater than 0");

#include <array>
#include <cmath>

struct NeuralNetwork {
    std::array<std::array<float, INPUT_SIZE>, HIDDEN_SIZE> W1;
    std::array<float, HIDDEN_SIZE> b1;
    std::array<std::array<float, HIDDEN_SIZE>, OUTPUT_SIZE> W2;
    std::array<float, OUTPUT_SIZE> b2;
};

void forward(NeuralNetwork& nn,
            const std::array<float, INPUT_SIZE>& x,
            std::array<float, OUTPUT_SIZE>& y) {

    std::array<float, HIDDEN_SIZE> hidden{};
    y = {};

    for(int i = 0;i < HIDDEN_SIZE; i++) {
        for(int j = 0;j < INPUT_SIZE; j++) {
            hidden[i] += x[j] * nn.W1[i][j];
        }
        hidden[i] += nn.b1[i];
        hidden[i] = activate(hidden[i]);
    }
    
    for(int i = 0;i < OUTPUT_SIZE; i++) {
        for(int j = 0;j < HIDDEN_SIZE; j++) {
            y[i] += hidden[j] * nn.W2[i][j];
        }
        y[i] += nn.b2[i];
        y[i] = activate(y[i]);
    }
}

void forward(NeuralNetwork& nn,
            const std::array<float, INPUT_SIZE>& x,
            std::array<float, OUTPUT_SIZE>& y,
            std::array<float, HIDDEN_SIZE>& h,
            const std::array<float, OUTPUT_SIZE>& target,
            float& error){

    h = {};
    y = {};
    error = 0;

    for(int i = 0;i < HIDDEN_SIZE; i++) {
        for(int j = 0;j < INPUT_SIZE; j++) {
            h[i] += x[j] * nn.W1[i][j];
        }
        h[i] += nn.b1[i];
        h[i] = activate(h[i]);
    }
    
    for(int i = 0;i < OUTPUT_SIZE; i++) {
        for(int j = 0;j < HIDDEN_SIZE; j++) {
            y[i] += h[j] * nn.W2[i][j];
        }
        y[i] += nn.b2[i];
        y[i] = activate(y[i]);
        error += (y[i] - target[i]) * (y[i] - target[i]);
    }

    error /= OUTPUT_SIZE;
}

float activate(float x) {
    return std::tanh(x);
}

float activate_derivative(float x) {
    return 1 - (std::tanh(x) * std::tanh(x));
}