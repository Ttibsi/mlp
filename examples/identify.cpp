#include <cstdint>
#include <filesystem>
#include <string>
#include <print>

#include "../mlp.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"

#include "raylib.h"

namespace fs = std::filesystem;

struct Mnist_Image {
    int height = 0;
    int width = 0;
    int bpp = 0;
    std::uint8_t* data;
    int img_value;

    constexpr Mnist_Image(const std::string& img_file, int value) {
        data = stbi_load(img_file.c_str(), &height, &width, &bpp, 3);
        img_value = value;
    }

    constexpr std::vector<ValuePtr_t> toValues() {
        std::vector<ValuePtr_t> ret = {};
        ret.reserve(height*width);

        for (int i = 0; i < height * width; i++) {
            ret.push_back(Value::Create(data[i]));
        }

        return ret;
    }

    constexpr std::vector<ValuePtr_t> expectedValue() {
        std::vector<ValuePtr_t> ret = {};
        ret.reserve(10);

        for (int i = 0; i < 10; i++) {
            if (i == img_value) { ret.push_back(Value::Create(0.0)); }
            else { ret.push_back(Value::Create(1.0)); }
        }

        return ret;
    }

    ~Mnist_Image() {
        stbi_image_free(data);
    }
};

int main() {
    const std::size_t img_size = 28*28;
    MLP mlp = MLP(img_size, {16, 16, 10});

    // step 1: load images from mnist dataset
    std::vector<std::vector<ValuePtr_t>> inputs = {};
    std::vector<std::vector<ValuePtr_t>> expected = {};

    for (const auto& digit_dir : fs::directory_iterator("mnist")) {
        if (digit_dir.is_directory()) {
            int digit = std::stoi(digit_dir.path().filename().string());
            for (const auto& entry : fs::directory_iterator(digit_dir.path())) {
                if (entry.is_regular_file()) {
                    Mnist_Image i = Mnist_Image(entry.path().string(), digit);
                    inputs.push_back(i.toValues());
                    expected.push_back(i.expectedValue());

                    std::println("Trained: {}", entry.path().string());
                }
            }
        }
    }

    std::println("Inputs: {}, outputs: {}", inputs.size(), expected.size());
    mlp.setDimensions(inputs, expected);

    // step 2: train an mlp on each image
    const int iterations = 5;
    const float rate = 0.1;
    std::vector<std::vector<ValuePtr_t>> result = mlp.gradientDescent(rate, iterations);

    // step 3: open a gui with:
        // 28x28 "pad" to draw in 
        // Visual representation of the mlp nodes (or just the output layer)
}
