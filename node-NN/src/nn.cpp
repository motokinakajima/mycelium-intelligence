constexpr int INPUT_SIZE = 7;
constexpr int HIDDEN_SIZE = 6;
constexpr int OUTPUT_SIZE = 5;
constexpr float LEARNING_RATE = 0.1f;

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
    error = 0.0f;

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

    error /= (OUTPUT_SIZE * 2);
}

void back_propagate(NeuralNetwork& nn,
        const std::array<float, INPUT_SIZE>& x,
        const std::array<float, OUTPUT_SIZE>& target) {

    std::array<float, HIDDEN_SIZE> h;
    std::array<float, OUTPUT_SIZE> y;
    float error;

    forward(nn, x, y, h, target, error);
    std::array<float, OUTPUT_SIZE> output_deltas{};
    std::array<float, HIDDEN_SIZE> hidden_deltas{};

    for(int i = 0;i < OUTPUT_SIZE; i++) {
        float output_error = target[i] - y[i];
        output_deltas[i] = output_error * (1 - y[i] * y[i]);
        output_deltas[i] *= LEARNING_RATE;
    }
    for(int i = 0;i < HIDDEN_SIZE; i++) {
        float hidden_error = 0.0f;
        for(int j = 0;j < OUTPUT_SIZE; j++) {
            hidden_error += output_deltas[j] * nn.W2[j][i];
        }
        hidden_deltas[i] = hidden_error * (1 - h[i] * h[i]);
        hidden_deltas[i] *= LEARNING_RATE;
    }

    for(int i = 0;i < OUTPUT_SIZE; i++) {
        for(int j = 0;j < HIDDEN_SIZE; j++) {
            nn.W2[i][j] += output_deltas[i] * h[j];
        }
        nn.b2[i] += output_deltas[i];
    }
    for(int i = 0;i < HIDDEN_SIZE; i++) {
        for(int j = 0;j < INPUT_SIZE; j++) {
            nn.W1[i][j] += hidden_deltas[i] * x[j];
        }
        nn.b1[i] += hidden_deltas[i];
    }
}


float activate(float x) {
    return std::tanh(x);
}