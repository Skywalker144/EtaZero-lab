#pragma once
#include "inference.h"
#include <random>
namespace etazero {
// Stateless per-observation outputs: reproducible across batch/thread scheduling.
class RandomBackend final : public Backend {
    int actions_;
    uint64_t seed_;
public:
    RandomBackend(int canvas, uint64_t seed);
    std::vector<Evaluation> evaluate(const InferenceInputs& inputs) override;
};
}
