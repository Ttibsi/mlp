#include <print>
#include "mlp.h"

int main() {
    std::vector<std::vector<ValuePtr_t>> inputs = {
        {Value::Create(0.0), Value::Create(0.0)},
        {Value::Create(1.0), Value::Create(0.0)},
        {Value::Create(0.0), Value::Create(1.0)},
        {Value::Create(1.0), Value::Create(1.0)},
    };

    std::vector<std::vector<ValuePtr_t>> expected = {
        {Value::Create(0.0)},
        {Value::Create(1.0)},
        {Value::Create(1.0)},
        {Value::Create(0.0)}
    };

    MLP mlp = MLP(2, {2, 1});
    mlp.setDimensions(inputs, expected);
    std::vector<std::vector<ValuePtr_t>> result = mlp.gradientDescent(0.1, 15000);
    std::println("{}", result);
}
