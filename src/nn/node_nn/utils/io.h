#pragma once

#include "../nn.h"
#include <string>

namespace node_nn {

    bool save_model(const std::string &filename, const NeuralNetwork &nn);

    bool load_model(const std::string &filename, const NeuralNetwork &nn);

}