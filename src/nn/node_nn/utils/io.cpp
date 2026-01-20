#include "../nn.h"
#include <iostream>
#include <fstream>
#include <string>

namespace node_nn {

    bool save_model(const std::string &filename, const NeuralNetwork &nn) {
        std::string full_path = filename;
        if (full_path.find('.') == std::string::npos) {
            full_path += ".nn";
        }

        std::ofstream ofs(full_path, std::ios::binary);
        if (!ofs) {
            std::cerr << "Error: Could not open file for writing: " << full_path << std::endl;
            return false;
        }

        ofs.write(reinterpret_cast<const char*>(&nn), sizeof(Parameters));

        return ofs.good();
    }

    bool load_model(const std::string &filename, NeuralNetwork &nn) {
        std::string full_path = filename;
        if (full_path.find('.') == std::string::npos) {
            full_path += ".nn";
        }

        std::ifstream ifs(full_path, std::ios::binary);
        if (!ifs) {
            std::cerr << "Error: Could not open file for reading: " << full_path << std::endl;
            return false;
        }

        ifs.seekg(0, std::ios::end);
        if (ifs.tellg() != sizeof(Parameters)) {
            std::cerr << "Error: File size mismatch in " << full_path << std::endl;
            return false;
        }
        ifs.seekg(0, std::ios::beg);

        ifs.read(reinterpret_cast<char*>(&nn), sizeof(Parameters));

        return ifs.good();
    }

}