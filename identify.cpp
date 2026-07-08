#include <cstdint>
#include <filesystem>
#include <string>

#include "mlp.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace fs = std::filesystem;

struct Image {
    int height = 0;
    int width = 0;
    int bpp = 0;
    std::uint8_t* data;
    int img_value;

    constexpr Image(const std::string& img_file, int value) {
        data = stbi_load(img_file.c_str(), &height, &width, &bpp, 3);
        img_value = value;
    }

    constexpr std::vector<ValuePtr_t> toValues() {
        std::vector<ValuePtr_t> ret = {};

        return ret;
    }

    constexpr std::vector<ValuePtr_t> expectedValue() {
        std::vector<ValuePtr_t> ret = {};

        return ret;
    }

    ~Image() {
        stbi_image_free(data);
    }
};

int main() {
    const std::size_t img_size = 28*28;
    MLP mlp = MLP(img_size, {16, 16, 10});

    // step 1: load images from mnist dataset
    std::vector<std::vector<ValuePtr_t>> inputs = {};
    std::vector<std::vector<ValuePtr_t>> expected = {};

    for (const auto& entry : fs::directory_iterator("mnist")) {
        if (entry.is_regular_file()) {
            Image i = Image(entry.path().filename(), 0);
            inputs.push_back(i.toValues());
            expected.push_back(i.expectedValue());
        }
    }

    mlp.setDimensions(inputs, expected);

    // step 2: train an mlp on each image
    const int iterations = 1000;
    const float rate = 0.1;
    std::vector<std::vector<ValuePtr_t> result = mlp.gradientDescent(rate, iterations);

    // step 3: open a gui with:
        // 28x28 "pad" to draw in 
        // Visual representation of the mlp nodes (or just the output layer)
}
