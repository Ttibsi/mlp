#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <functional>
#include <memory>
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

struct Value;
using ValuePtr_t = std::shared_ptr<Value>;

struct Value: std::enable_shared_from_this<Value> {
    float data;
    float grad = 0.0;
    std::vector<ValuePtr_t> prev;
    Op op = Op::NONE;
    std::function<void()> backward = [](){};

    explicit Value(float f): data(f), prev({}) {}
    Value(float f, std::vector<ValuePtr_t> children, Op o):
        data(f), prev(std::move(children)), op(o) {}

    static ValuePtr_t Create(float f) {
        return std::make_shared<Value>(f);
    }

    static ValuePtr_t Random(float min, float max) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(min, max);
        return Create(dist(gen));
    }

    [[nodiscard]] ValuePtr_t add(const ValuePtr_t& other) {
        auto self = shared_from_this();
        auto out = std::make_shared<Value>(
            data + other->data,
            std::vector<ValuePtr_t>{self, other},
            Op::ADD
        );

        out->backward = [self, other, out]() mutable {
            self->grad += 1.0f * out->grad;
            other->grad += 1.0f * out->grad;
        };

        return out;
    }

    [[nodiscard]] ValuePtr_t add(float f) {
        return add(Create(f));
    }

    [[nodiscard]] ValuePtr_t sub(float f) {
        auto self = shared_from_this();
        auto tmp = std::make_shared<Value>(-f, std::vector<ValuePtr_t>{self}, Op::SUB);
        return add(tmp);
    }

    [[nodiscard]] ValuePtr_t mul(const ValuePtr_t& other) {
        auto self = shared_from_this();
        auto out = std::make_shared<Value>(
            data * other->data,
            std::vector<ValuePtr_t>{self, other},
            Op::MUL
        );

        out->backward = [self, other, out]() mutable {
            self->grad += other->data * out->grad;
            other->grad += self->data * out->grad;
        };
        return out;
    }

    [[nodiscard]] ValuePtr_t mul(float f) {
        return mul(Create(f));
    }

    [[nodiscard]] constexpr bool operator==(const Value& other) const {
        return data == other.data && grad == other.grad;
    }

    [[nodiscard]] ValuePtr_t div(const ValuePtr_t& other) {
        auto self = shared_from_this();
        auto out = std::make_shared<Value>(
            data / other->data,
            std::vector<ValuePtr_t>{self, other},
            Op::DIV
        );
        return out;
    }

    [[nodiscard]] ValuePtr_t div(float f) {
        return div(Create(f));
    }


    [[nodiscard]] ValuePtr_t activate() {
        auto self = shared_from_this();
        auto out = std::make_shared<Value>(
            sigmoidf(data),
            std::vector<ValuePtr_t>{self},
            Op::ACTIVATE
        );

        out->backward = [self, out]() mutable {
            self->grad += sigmoidfDerivative(out->grad);
        };

        return out;
    }

    void backprop() {
        std::vector<ValuePtr_t> topo = {};
        std::vector<ValuePtr_t> visited = {};

        std::function<void(const ValuePtr_t&)> impl = [&](const ValuePtr_t& v) {
            if (v && std::find(visited.begin(), visited.end(), v) == visited.end()) {
                visited.push_back(v);

                for (const auto& child : v->prev) {
                    impl(child);
                }

                topo.push_back(v);
            }
        };

        impl(shared_from_this());
        grad = 1.0;

        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->backward();
        }
    }
};

struct Neuron {
    std::vector<ValuePtr_t> weights = {};
    ValuePtr_t bias;

    explicit Neuron(int inputCount) : bias(Value::Random(-1, 1)) {
        for (int i = 0; i < inputCount; i++) {
            weights.push_back(Value::Random(-1, 1));
        }
    }

    [[nodiscard]] ValuePtr_t operator()(const std::vector<ValuePtr_t>& inputs) {
        assert(inputs.size() == weights.size());

        auto activation = bias;
        for (auto [wi, xi] : std::views::zip(weights, inputs)) {
            auto calc = wi->mul(xi);
            activation = activation->add(calc);
        }

        return activation->activate();
    }
};

struct Layer {
    std::vector<Neuron> neurons = {};

    Layer(int inputs, int outputs) {
        for (int i = 0; i < outputs; i++) { neurons.push_back(Neuron(inputs)); }
    }

    [[nodiscard]] std::vector<ValuePtr_t> operator()(const std::vector<ValuePtr_t>& inputs) {
        std::vector<ValuePtr_t> outs = {};
        std::for_each(neurons.begin(), neurons.end(), [&](Neuron& n) mutable {
                outs.push_back(n(inputs)); 
                });
        return outs;
    }
};

struct MLP {
    std::vector<Layer> layers = {};

    MLP(int inputs, std::vector<int> outputs) {
        std::vector sz = {inputs};
        sz.insert(sz.end(), outputs.begin(), outputs.end());

        for (std::size_t i = 0; i < outputs.size(); i++) {
            layers.push_back(Layer(sz.at(i), sz.at(i + 1)));
        }
    }

    [[nodiscard]] std::vector<ValuePtr_t> operator()(std::vector<ValuePtr_t> inputs) {
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

