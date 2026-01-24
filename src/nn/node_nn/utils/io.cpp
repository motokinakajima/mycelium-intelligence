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

    bool load_training_data(const std::string &filename, TrainingData &data) {
                auto trim = [](std::string &s) {
                    s.erase(0, s.find_first_not_of(" \t"));
                    s.erase(s.find_last_not_of(" \t") + 1);
                };
        std::ifstream ifs(filename);
        if (!ifs) {
            std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
            return false;
        }

        data.input.clear();
        data.target.clear();
        std::string line;
        size_t line_num = 0;
        while (std::getline(ifs, line)) {
            ++line_num;
            if (line.empty() || line[0] == '#') continue;

            std::vector<float> values;
            size_t start = 0, end = 0;
            while ((end = line.find(',', start)) != std::string::npos) {
                std::string token = line.substr(start, end - start);
                trim(token);
                try {
                    values.push_back(std::stof(token));
                } catch (...) {
                    std::cerr << "Error: Invalid float value in CSV at line " << line_num << std::endl;
                    return false;
                }
                start = end + 1;
            }
            
            if (start < line.size()) {
                std::string token = line.substr(start);
                trim(token);
                try {
                    values.push_back(std::stof(token));
                } catch (...) {
                    std::cerr << "Error: Invalid float value in CSV at line " << line_num << std::endl;
                    return false;
                }
            }

            if (values.size() != INPUT_SIZE + OUTPUT_SIZE) {
                std::cerr << "Error: CSV row does not match INPUT_SIZE + OUTPUT_SIZE at line " << line_num << std::endl;
                return false;
            }

            std::array<float, INPUT_SIZE> x{};
            std::array<float, OUTPUT_SIZE> y{};
            for (int i = 0; i < INPUT_SIZE; ++i) x[i] = values[i];
            for (int i = 0; i < OUTPUT_SIZE; ++i) y[i] = values[INPUT_SIZE + i];
            data.input.push_back(x);
            data.target.push_back(y);
        }
        return true;
    }

}