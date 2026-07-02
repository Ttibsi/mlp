#include <print>
#include "mlp.h"

int main() {
    std::vector<std::vector<ValuePtr_t>> xs = {
        {Value::Create(2.0), Value::Create(3.0), Value::Create(-1.0)},
        {Value::Create(3.0), Value::Create(-1.0), Value::Create(0.5)},
        {Value::Create(0.5), Value::Create(1.0), Value::Create(1.0)},
        {Value::Create(1.0), Value::Create(1.0), Value::Create(-1.0)}
    };

    std::vector<Value> ys = {Value(1.0), Value(-1.0), Value(-1.0), Value(1.0)};

    MLP n = MLP(3, {4,4,1});
    std::vector<std::vector<ValuePtr_t>> ypreds = {};
    for (auto x: xs) {
        ypreds.push_back(n(x));
    }

    ValuePtr_t avg_loss = n.calcLosses(ys, ypreds);
    avg_loss->backprop();
    std::println("avg loss = {}", avg_loss);

    auto v = n.layers.at(0).neurons.at(0).weights.at(0);
    std::println("v = {}", v);
}
