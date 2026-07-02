#include <print>
#include "mlp.h"

int main() {
    const int iterations = 1;

    std::vector<std::vector<ValuePtr_t>> inputs = {
        {Value::Create(0.0), Value::Create(0.0)},
        {Value::Create(1.0), Value::Create(0.0)},
        {Value::Create(0.0), Value::Create(1.0)},
        {Value::Create(1.0), Value::Create(1.0)},
    };

    std::vector<Value> expected = {Value(0), Value(1), Value(1), Value(1)};

    MLP mlp = MLP(2, {2, 1});
    for (int i = 0; i < iterations; i++) {
        std::vector<std::vector<ValuePtr_t>> outputs = {};
        for (auto x: inputs) {
            outputs.push_back(mlp(x));
        }

        std::vector<Value> losses = mlp.calcLosses(expected, outputs);
        std::println("{}", losses);
    }
}
