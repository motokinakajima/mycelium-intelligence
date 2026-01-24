#include "../nn.h"
#include <cmath>
#include <vector>
#include <random>

namespace node_nn {
    void Parameters::set_zero() {
        for (auto& row : W1) row.fill(0.0f);
        b1.fill(0.0f);
        for (auto& row : W2) row.fill(0.0f);
        b2.fill(0.0f);
    }

    void Parameters::randomize() {
        std::random_device rd;
        auto gen = std::mt19937(rd());
        auto dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);

        for (auto& row : W1) {
            for (auto& w : row) w = dist(gen);
        }
        for (auto& b : b1) b = dist(gen);

        for (auto& row : W2) {
            for (auto& w : row) w = dist(gen);
        }
        for (auto& b : b2) b = dist(gen);
    }

    NeuralNetwork::NeuralNetwork() : Parameters() {
        this->randomize();
    }

    Gradients::Gradients() : Parameters() {
        this->set_zero();
    }

    AdamM::AdamM() : Parameters() {
        this->set_zero();
    }

    AdamV::AdamV() : Parameters() {
        this->set_zero();
    }

    AdamState::AdamState() : m(), v(), t(0) {}

    void forward(const NeuralNetwork &nn,
                 const std::array<float, INPUT_SIZE> &x,
                 std::array<float, OUTPUT_SIZE> &y,
                 std::array<float, HIDDEN_SIZE> &h) {

        h = {};
        y = {};

        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                h[i] += x[j] * nn.W1[i][j];
            }
            h[i] += nn.b1[i];
            h[i] = activate(h[i]);
        }

        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                y[i] += h[j] * nn.W2[i][j];
            }
            y[i] += nn.b2[i];
            y[i] = activate(y[i]);
        }
    }

    void forward(const NeuralNetwork &nn,
                 const std::array<float, INPUT_SIZE> &x,
                 std::array<float, OUTPUT_SIZE> &y) {
        std::array<float, HIDDEN_SIZE> h = {};
        forward(nn, x, y, h);
    }

    void cost(const std::array<float, OUTPUT_SIZE> &y,
              const std::array<float, OUTPUT_SIZE> &target,
              float &error) {
        error = 0.0f;
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            error += (y[i] - target[i]) * (y[i] - target[i]);
        }
        error /= (OUTPUT_SIZE * 2);
    }

    void single_back_propagate(NeuralNetwork &nn,
                               const std::array<float, INPUT_SIZE> &x,
                               const std::array<float, OUTPUT_SIZE> &target) {

        Gradients gradient;
        add_gradients(nn, x, target, gradient);
        apply_gradients(nn, gradient, 1.0f);
    }

    void back_propagate(NeuralNetwork &nn,
                        const std::vector<std::array<float, INPUT_SIZE>> &input,
                        const std::vector<std::array<float, OUTPUT_SIZE>> &target) {

        if (input.size() != target.size()) {
            return;
        }

        Gradients gradient;
        for (int i = 0;i < input.size(); i++) {
            add_gradients(nn, input[i], target[i], gradient);
        }
        apply_gradients(nn, gradient, static_cast<float>(input.size()));
    }

    void back_propagate(NeuralNetwork &nn, const TrainingData &data) {
        back_propagate(nn, data.input, data.target);
    }

    void adam(NeuralNetwork &nn,
              const std::vector<std::array<float, INPUT_SIZE>> &input,
              const std::vector<std::array<float, OUTPUT_SIZE>> &target,
              AdamState &state) {

        if (input.size() != target.size()) {
            return;
        }

        Gradients gradient;

        for (int i = 0;i < input.size(); i++) {
            add_gradients(nn, input[i], target[i], gradient);
        }

        average_gradients(gradient, static_cast<float>(input.size()));

        state.t += 1;

        float bias_correction_1 = 1.0f - std::pow(BETA_1, static_cast<float>(state.t));
        float bias_correction_2 = 1.0f - std::pow(BETA_2, static_cast<float>(state.t));

        const float lr = LEARNING_RATE;
        const float beta_1_complement = 1 - BETA_1;
        const float beta_2_complement = 1 - BETA_2;

        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                state.m.W1[i][j] = BETA_1 * state.m.W1[i][j] + beta_1_complement * gradient.W1[i][j];
                state.v.W1[i][j] = BETA_2 * state.v.W1[i][j] + beta_2_complement * gradient.W1[i][j] * gradient.W1[i][j];
                float m_hat = state.m.W1[i][j] / bias_correction_1;
                float v_hat = state.v.W1[i][j] / bias_correction_2;
                nn.W1[i][j] = nn.W1[i][j] + lr * (m_hat / std::sqrt(v_hat + EPSILON));
            }
            state.m.b1[i] = BETA_1 * state.m.b1[i] + beta_1_complement * gradient.b1[i];
            state.v.b1[i] = BETA_2 * state.v.b1[i] + beta_2_complement * gradient.b1[i] * gradient.b1[i];
            float m_hat = state.m.b1[i] / bias_correction_1;
            float v_hat = state.v.b1[i] / bias_correction_2;
            nn.b1[i] = nn.b1[i] + lr * (m_hat / std::sqrt(v_hat + EPSILON));
        }
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                state.m.W2[i][j] = BETA_1 * state.m.W2[i][j] + beta_1_complement * gradient.W2[i][j];
                state.v.W2[i][j] = BETA_2 * state.v.W2[i][j] + beta_2_complement * gradient.W2[i][j] * gradient.W2[i][j];
                float m_hat = state.m.W2[i][j] / bias_correction_1;
                float v_hat = state.v.W2[i][j] / bias_correction_2;
                nn.W2[i][j] = nn.W2[i][j] + lr * (m_hat / std::sqrt(v_hat + EPSILON));
            }
            state.m.b2[i] = BETA_1 * state.m.b2[i] + beta_1_complement * gradient.b2[i];
            state.v.b2[i] = BETA_2 * state.v.b2[i] + beta_2_complement * gradient.b2[i] * gradient.b2[i];
            float m_hat = state.m.b2[i] / bias_correction_1;
            float v_hat = state.v.b2[i] / bias_correction_2;
            nn.b2[i] = nn.b2[i] + lr * (m_hat / std::sqrt(v_hat + EPSILON));
        }
    }

    void adam(NeuralNetwork &nn,
              const TrainingData &data,
              AdamState &state) {
        adam(nn, data.input, data.target, state);
    }

    void add_gradients(const NeuralNetwork &nn,
                       const std::array<float, INPUT_SIZE> &x,
                       const std::array<float, OUTPUT_SIZE> &target,
                       Gradients &gradient) {

        std::array<float, HIDDEN_SIZE> h = {};
        std::array<float, OUTPUT_SIZE> y = {};
        float error;

        forward(nn, x, y, h);
        cost(y, target, error);
        std::array<float, OUTPUT_SIZE> output_deltas{};
        std::array<float, HIDDEN_SIZE> hidden_deltas{};
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            float output_error = target[i] - y[i];
            output_deltas[i] = output_error * (1 - y[i] * y[i]);
        }
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            float hidden_error = 0.0f;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                hidden_error += output_deltas[j] * nn.W2[j][i];
            }
            hidden_deltas[i] = hidden_error * (1 - h[i] * h[i]);
        }
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                gradient.W2[i][j] += output_deltas[i] * h[j];
            }
            gradient.b2[i] += output_deltas[i];
        }
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                gradient.W1[i][j] += hidden_deltas[i] * x[j];
            }
            gradient.b1[i] += hidden_deltas[i];
        }
    }

    void apply_gradients(NeuralNetwork &nn, const Gradients &gradient, float batch_size) {
        float lr = LEARNING_RATE / batch_size;
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                nn.W1[i][j] += gradient.W1[i][j] * lr;
            }
            nn.b1[i] += gradient.b1[i] * lr;
        }
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                nn.W2[i][j] += gradient.W2[i][j] * lr;
            }
            nn.b2[i] += gradient.b2[i] * lr;
        }
    }

    void average_gradients(Gradients &g, const float batch_size) {
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                g.W1[i][j] /= batch_size;
            }
            g.b1[i] /= batch_size;
        }
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                g.W2[i][j] /= batch_size;
            }
            g.b2[i] /= batch_size;
        }
    }

    void separate_train_data(TrainingData &learn, TrainingData &test, float test_ratio) {
        size_t total_size = learn.input.size();
        size_t test_size = static_cast<size_t>(total_size * test_ratio);

        test.input.insert(test.input.end(), learn.input.end() - test_size, learn.input.end());
        test.target.insert(test.target.end(), learn.target.end() - test_size, learn.target.end());

        learn.input.resize(total_size - test_size);
        learn.target.resize(total_size - test_size);
    }

    float activate(float x) {
        return std::tanh(x);
    }

}