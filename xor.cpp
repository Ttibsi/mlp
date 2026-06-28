#include <print>
#include "mlp.h"

int main() {
    const int iterations = 1;

    MLP mlp = MLP(2, {2, 1});
    for (int i = 0; i < iterations; i++) {
        for (auto v: mlp({1.0, 0.0})) {
            std::println("{}", v);
            v.backprop();
        }
    }

}
