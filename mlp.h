#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <functional>
#include <print>
#include <random>
#include <ranges>
#include <stack>
#include <vector>

enum struct Op {
    ADD,
    SUB,
    MUL,
    DIV,
    ACTIVATE,
    NONE
};

[[nodiscard]] constexpr float sigmoidf(const float x) {
    return 1.f / (1.f + std::expf(x));
}

[[nodiscard]] constexpr float sigmoidfDerivative(const float x) {
    return x * (1 - x);
}

struct Value {
    float data;
    float grad = 0.0;
    std::vector<Value*> prev;
    Op op = Op::NONE;
    std::function<void()> backward = [](){};

    constexpr Value(float f): data(f), prev({}) {}
    constexpr Value(float f, std::vector<Value*> children, Op o):
        data(f), prev(children), op(o) {}

    static constexpr Value Random(float min, float max) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> dist(min, max);

        return Value(dist(gen));
    }

    [[nodiscard]] constexpr Value operator+(Value& other) {
        Value out = Value(data + other.data, {this, &other}, Op::ADD);

        out.backward = [=, this]() mutable {
            this->grad += 1.0 * out.grad;
            other.grad += 1.0 * out.grad;
        };

        return out;
    }

    [[nodiscard]] constexpr Value operator+(float f) {
        Value tmp = Value(f);
        return *this + tmp;
    }

    constexpr void operator+=(Value& other) {
        *this = *this + other;
    }

    [[nodiscard]] constexpr Value operator-(float f) {
        Value tmp = Value(-f, {this}, Op::SUB);
        return *this + tmp;
    }

    [[nodiscard]] constexpr Value operator*(Value& other) {
        Value out = Value(data * other.data, {this, &other}, Op::MUL);

        out.backward = [=, this]() mutable {
            this->grad += other.data * out.grad;
            other.grad += this->data * out.grad;
        };
        return out;
    }

    [[nodiscard]] constexpr Value operator*(float f) {
        Value tmp = Value(f);
        return *this * tmp;
    }

    [[nodiscard]] constexpr bool operator==(const Value& other) {
        return data == other.data && grad == other.grad;
    }

    [[nodiscard]] constexpr Value operator/(Value& other) {
        Value out = Value(data / other.data, {this, &other}, Op::DIV);
        return out;
    }

    [[nodiscard]] constexpr Value operator/(float f) {
        Value prev = Value(f);
        return *this / prev;
    }


    [[nodiscard]] constexpr Value activate() {
        Value out = Value(sigmoidf(data), {this}, Op::ACTIVATE);

        out.backward = [=, this]() mutable {
            this->grad += sigmoidfDerivative(out.grad);
        };

        return out;
    }

    void backprop() {
        std::vector<Value*> topo = {};
        std::vector<Value*> visited = {};

        std::function<void(Value*)> impl = [&](Value* v) {
            if (v && std::find(visited.begin(), visited.end(), v) == visited.end()) {
                visited.push_back(v);

                for (auto child : v->prev) {
                    impl(child);
                }

                topo.push_back(v);
            }
        };

        impl(this);
        grad = 1.0;

        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->backward();
        }
}



};

struct Neuron {
    std::vector<Value> weights = {};
    Value bias;

    [[nodiscard]] constexpr Neuron(int inputCount) : bias(Value::Random(-1, 1)) {
        for (int i = 0; i < inputCount; i++) {
            weights.push_back(Value::Random(-1, 1));
        }
    }

    [[nodiscard]] constexpr Value operator()(std::vector<Value> inputs) {
        assert(inputs.size() == weights.size());

        Value activation = bias;
        for (auto [wi, xi] : std::views::zip(weights, inputs)) {
            Value calc = wi*xi;
            activation += calc;
        }

        return activation.activate();
    }
};

struct Layer {
    std::vector<Neuron> neurons = {};

    [[nodiscard]] constexpr Layer(int inputs, int outputs) {
        for (int i = 0; i < outputs; i++) { neurons.push_back(Neuron(inputs)); }
    }

    [[nodiscard]] constexpr std::vector<Value> operator()(std::vector<Value> inputs) {
        std::vector<Value> outs = {};
        std::for_each(neurons.begin(), neurons.end(), [&](Neuron& n) mutable {
            outs.push_back(n(inputs)); 
        });
        return outs;
    }
};

struct MLP {
    std::vector<Layer> layers = {};

    [[nodiscard]] constexpr MLP(int inputs, std::vector<int> outputs) {
        std::vector sz = {inputs};
        sz.insert(sz.end(), outputs.begin(), outputs.end());

        for (std::size_t i = 0; i < outputs.size(); i++) {
            layers.push_back(Layer(sz.at(i), sz.at(i + 1)));
        }
    }

    [[nodiscard]] constexpr std::vector<Value> operator()(std::vector<Value> inputs) {
        for (auto&& l: layers) {
            inputs = l(inputs);
        }

        return inputs;
    }
};

template <>
struct std::formatter<Value> {
    constexpr auto parse(std::format_parse_context& ctx) const {
          return ctx.begin(); 
    }
    constexpr auto format(const Value& v, std::format_context& ctx) const {
        return std::format_to(ctx.out(), "Value({:.5f}, {:.5f})", v.data, v.grad);
    }
};
