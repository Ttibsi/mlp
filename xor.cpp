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
    mlp.setDimensions(inputs, expected);
    std::vector<ValuePtr_t> result = mlp.gradientDescent(0.5, 300);
    std::println("{}", result);
}
