#include <print>
#include "../mlp.h"

int main() {
    std::vector<std::vector<ValuePtr_t>> xs = {
        {Value::Create(2.0), Value::Create(3.0), Value::Create(-1.0)},
        {Value::Create(3.0), Value::Create(-1.0), Value::Create(0.5)},
        {Value::Create(0.5), Value::Create(1.0), Value::Create(1.0)},
        {Value::Create(1.0), Value::Create(1.0), Value::Create(-1.0)}
    };

    std::vector<std::vector<ValuePtr_t>> ys = {
        {Value::Create(1.0)},
        {Value::Create(0.0)},
        {Value::Create(0.0)},
        {Value::Create(1.0)}
    };

    MLP n = MLP(3, {4,4,1});
    // std::vector<std::vector<ValuePtr_t>> ypreds = {};
    // for (auto x: xs) {
    //     ypreds.push_back(n(x));
    // }
    //
    // ValuePtr_t loss_rate = n.calcLosses(ys, ypreds);
    // loss_rate->backprop();
    // // std::println("avg loss = {}", loss_rate);
    //
    // ValuePtr_t v = n.layers.at(0).neurons.at(0).weights.at(0);
    // std::println("v = {}", v);

    n.setDimensions(xs, ys);
    auto ret = n.gradientDescent(0.01, 500);
    std::println("{}", ret);
    // v = n.layers.at(0).neurons.at(0).weights.at(0);
    // std::println("v = {}", v);
}
