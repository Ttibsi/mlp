#include <print>
#include "mlp.h"

int main() {
    const int iterations = 1;

    MLP mlp = MLP(2, {2, 1});
    for (int i = 0; i < iterations; i++) {
        auto outputs = mlp({Value::Create(1.0f), Value::Create(0.0f)});
        for (const auto& v : outputs) {
            std::println("{}", *v);
            v->backprop();
        }
    }
}

